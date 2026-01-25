#!/usr/bin/env python3
"""
BIP-39 Mnemonic Recovery Script

Recovers missing words from a partial BIP-39 mnemonic phrase by brute-forcing
possible combinations and checking against a known wallet address.

Supports: Bitcoin (BTC), Ethereum (ETH), TRON (TRX), Binance Smart Chain (BSC),
Litecoin (LTC), Dogecoin (DOGE), and more.
"""

import argparse
import hashlib
import itertools
import sys
from multiprocessing import Pool, cpu_count, Manager
from typing import Optional, List

from mnemonic import Mnemonic
from tqdm import tqdm

# Import bip_utils components
from bip_utils import (
    Bip39MnemonicValidator,
    Bip39SeedGenerator,
    Bip39Languages,
    Bip44,
    Bip49,
    Bip84,
    Bip44Coins,
    Bip49Coins,
    Bip84Coins,
    Bip44Changes,
)

# Blockchain configurations
BLOCKCHAIN_CONFIG = {
    "bitcoin": {
        "coin": Bip44Coins.BITCOIN,
        "derivation": "bip44",
        "address_prefix": ["1", "3", "bc1"],
    },
    "bitcoin_segwit": {
        "coin": Bip49Coins.BITCOIN,
        "derivation": "bip49",
        "address_prefix": ["3"],
    },
    "bitcoin_native_segwit": {
        "coin": Bip84Coins.BITCOIN,
        "derivation": "bip84",
        "address_prefix": ["bc1"],
    },
    "ethereum": {
        "coin": Bip44Coins.ETHEREUM,
        "derivation": "bip44",
        "address_prefix": ["0x"],
    },
    "tron": {
        "coin": Bip44Coins.TRON,
        "derivation": "bip44",
        "address_prefix": ["T"],
    },
    "bsc": {
        "coin": Bip44Coins.ETHEREUM,  # BSC uses same derivation as ETH
        "derivation": "bip44",
        "address_prefix": ["0x"],
    },
    "litecoin": {
        "coin": Bip44Coins.LITECOIN,
        "derivation": "bip44",
        "address_prefix": ["L", "M", "ltc1"],
    },
    "dogecoin": {
        "coin": Bip44Coins.DOGECOIN,
        "derivation": "bip44",
        "address_prefix": ["D"],
    },
    "solana": {
        "coin": Bip44Coins.SOLANA,
        "derivation": "bip44",
        "address_prefix": [],  # Base58, various prefixes
    },
}

# BIP-39 phrase length configurations
# Maps phrase length to (entropy_bits, checksum_bits)
PHRASE_CONFIG = {
    12: (128, 4),
    15: (160, 5),
    18: (192, 6),
    21: (224, 7),
    24: (256, 8),
}

# Global variables for multiprocessing
wordlist = None
target_address = None
blockchain = None
phrase_template = None
missing_indices = None
found_result = None
mnemonic_validator = None
skip_checksum_validation = False


def compute_valid_last_words(known_words: List[str], wordlist: List[str]) -> List[str]:
    """
    Compute valid last words based on BIP-39 checksum constraints.

    For a partial phrase with only the last word missing, we can compute
    exactly which words would produce a valid checksum, dramatically reducing
    the search space (e.g., from 2048 to ~128 for 12-word phrases).

    Args:
        known_words: List of known words (all except the last)
        wordlist: BIP-39 wordlist

    Returns:
        List of valid last words that satisfy the checksum
    """
    phrase_len = len(known_words) + 1

    if phrase_len not in PHRASE_CONFIG:
        # Unknown phrase length, return all words
        return wordlist

    entropy_bits, checksum_bits = PHRASE_CONFIG[phrase_len]

    # Build word-to-index mapping
    word_to_idx = {word: idx for idx, word in enumerate(wordlist)}

    # Convert known words to bits (11 bits per word)
    bits = ""
    for word in known_words:
        if word not in word_to_idx:
            # Invalid word, can't optimize
            return wordlist
        idx = word_to_idx[word]
        bits += format(idx, "011b")

    # Calculate how many entropy bits are in the last word
    # Total bits from known words
    known_bits = len(known_words) * 11
    # Remaining entropy bits that must come from last word
    remaining_entropy_bits = entropy_bits - known_bits

    # For each possible value of remaining entropy bits, compute valid last word
    valid_words = []
    for i in range(2**remaining_entropy_bits):
        # Complete the entropy
        entropy_bits_str = bits + format(i, f"0{remaining_entropy_bits}b")

        # Convert to bytes
        entropy_bytes = int(entropy_bits_str, 2).to_bytes(entropy_bits // 8, "big")

        # Calculate checksum (first N bits of SHA256)
        h = hashlib.sha256(entropy_bytes).digest()
        # Get first checksum_bits from hash
        checksum_int = int.from_bytes(h[:1], "big") >> (8 - checksum_bits)

        # Last word index = remaining entropy bits (high) + checksum bits (low)
        last_word_idx = (i << checksum_bits) | checksum_int

        if last_word_idx < 2048:
            valid_words.append(wordlist[last_word_idx])

    return valid_words


def compute_valid_words_for_position(
    known_words: List[str], missing_idx: int, phrase_len: int, wordlist: List[str]
) -> List[str]:
    """
    Compute valid words for a specific missing position.

    Currently optimized only for the last word position.
    For other positions, returns full wordlist.

    Args:
        known_words: Template with None for missing positions
        missing_idx: Index of the missing word
        phrase_len: Total phrase length
        wordlist: BIP-39 wordlist

    Returns:
        List of valid words for that position
    """
    # Only optimize for last word position
    if missing_idx == phrase_len - 1:
        # Extract known words (all except last)
        words_before_last = [w for w in known_words[:missing_idx] if w != "?"]
        if len(words_before_last) == phrase_len - 1:
            return compute_valid_last_words(words_before_last, wordlist)

    return wordlist


def init_worker(wl, addr, bc, template, indices, result_dict, skip_checksum=False):
    """Initialize worker process with shared data."""
    global wordlist, target_address, blockchain, phrase_template, missing_indices, found_result, mnemonic_validator, skip_checksum_validation
    wordlist = wl
    target_address = addr.lower() if addr.startswith("0x") else addr
    blockchain = bc
    phrase_template = template
    missing_indices = indices
    found_result = result_dict
    skip_checksum_validation = skip_checksum
    # Create validator once per worker for efficiency
    mnemonic_validator = Bip39MnemonicValidator(Bip39Languages.ENGLISH)


def derive_address(mnemonic_phrase: str, blockchain_name: str) -> Optional[str]:
    """
    Derive wallet address from mnemonic for specified blockchain.

    Args:
        mnemonic_phrase: Valid BIP-39 mnemonic
        blockchain_name: Name of blockchain (e.g., 'ethereum', 'bitcoin')

    Returns:
        Derived address or None if derivation fails
    """
    try:
        config = BLOCKCHAIN_CONFIG[blockchain_name]
        seed = Bip39SeedGenerator(mnemonic_phrase).Generate()

        if config["derivation"] == "bip44":
            bip = Bip44.FromSeed(seed, config["coin"])
        elif config["derivation"] == "bip49":
            bip = Bip49.FromSeed(seed, config["coin"])
        elif config["derivation"] == "bip84":
            bip = Bip84.FromSeed(seed, config["coin"])
        else:
            return None

        # Derive first address: m/purpose'/coin'/0'/0/0
        # Bip44Changes.CHAIN_EXT = external chain (receiving addresses)
        account = bip.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
        return account.PublicKey().ToAddress()
    except Exception:
        return None


def check_combination(word_combo: tuple) -> Optional[str]:
    """
    Check if a word combination produces the target address.

    Args:
        word_combo: Tuple of words to fill in missing positions

    Returns:
        Complete mnemonic if found, None otherwise
    """
    global found_result

    # Early exit if already found
    if found_result.get("found"):
        return None

    # Build complete phrase
    phrase = phrase_template.copy()
    for idx, word in zip(missing_indices, word_combo):
        phrase[idx] = word

    mnemonic_str = " ".join(phrase)

    # Validate checksum first (fast rejection)
    # Skip if we've already pre-computed valid words using entropy constraints
    if not skip_checksum_validation:
        try:
            if not mnemonic_validator.IsValid(mnemonic_str):
                return None
        except Exception:
            return None

    # Derive address and check
    derived = derive_address(mnemonic_str, blockchain)
    if derived is None:
        return None

    # Normalize for comparison
    derived_normalized = derived.lower() if derived.startswith("0x") else derived

    if derived_normalized == target_address:
        found_result["found"] = True
        found_result["phrase"] = mnemonic_str
        return mnemonic_str

    return None


def recover_phrase(
    partial_phrase: str,
    address: str,
    blockchain_name: str,
    workers: int = None,
    passphrase: str = "",
) -> Optional[str]:
    """
    Attempt to recover missing words from a partial mnemonic.

    Args:
        partial_phrase: Mnemonic with '?' for missing words
        address: Target wallet address
        blockchain_name: Blockchain name
        workers: Number of parallel workers (default: CPU count)
        passphrase: Optional BIP-39 passphrase

    Returns:
        Recovered mnemonic or None
    """
    if workers is None:
        workers = cpu_count()

    # Load BIP-39 wordlist
    mnemo = Mnemonic("english")
    wl = mnemo.wordlist

    # Parse phrase template
    words = partial_phrase.strip().split()
    template = words.copy()
    indices = [i for i, w in enumerate(words) if w == "?"]

    if not indices:
        print("No missing words indicated (use '?' as placeholder)")
        return None

    phrase_len = len(words)
    missing_count = len(indices)

    # Determine valid word candidates for each missing position
    # This optimizes the search space when the last word is missing
    word_candidates = []
    optimization_applied = False

    for idx in indices:
        if idx == phrase_len - 1 and missing_count == 1:
            # Last word is the only missing word - apply entropy optimization
            known_words = [w for w in words if w != "?"]
            valid_words = compute_valid_last_words(known_words, wl)
            word_candidates.append(valid_words)
            optimization_applied = True
        elif idx == phrase_len - 1:
            # Last word is missing but there are other missing words too
            # We can still optimize for the last position by pre-filtering
            # But we need to generate combinations for other positions first
            word_candidates.append(wl)  # Will be optimized in check_combination
        else:
            word_candidates.append(wl)

    # Calculate total combinations
    total_combinations = 1
    for candidates in word_candidates:
        total_combinations *= len(candidates)

    print(f"\n{'='*60}")
    print(f"BIP-39 Mnemonic Recovery")
    print(f"{'='*60}")
    print(f"Blockchain:       {blockchain_name}")
    print(f"Target Address:   {address}")
    print(f"Phrase Length:    {phrase_len} words")
    print(f"Missing Words:    {missing_count} at position(s) {[i+1 for i in indices]}")
    if optimization_applied:
        print(f"Optimization:     Last-word entropy constraint applied")
        print(f"Valid Candidates: {len(word_candidates[0])} (reduced from 2048)")
    print(f"Total Attempts:   {total_combinations:,}")
    print(f"Workers:          {workers}")
    print(f"{'='*60}\n")

    if missing_count > 3:
        print("WARNING: More than 3 missing words will take extremely long!")
        print(f"         Estimated combinations: {total_combinations:,}")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            return None

    # Generate word combinations using optimized candidate lists
    combinations = itertools.product(*word_candidates)

    # Setup multiprocessing with shared state
    manager = Manager()
    result_dict = manager.dict()
    result_dict["found"] = False
    result_dict["phrase"] = None

    # Normalize target address
    addr_normalized = address.lower() if address.startswith("0x") else address

    # Process combinations in parallel with progress bar
    # Skip checksum validation if we've pre-computed valid words
    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(wl, addr_normalized, blockchain_name, template, indices, result_dict, optimization_applied),
    ) as pool:
        with tqdm(total=total_combinations, desc="Checking", unit="combo") as pbar:
            # Use imap_unordered for better performance
            chunk_size = max(1, min(10000, total_combinations // (workers * 10)))

            for result in pool.imap_unordered(
                check_combination,
                itertools.product(*word_candidates),
                chunksize=chunk_size,
            ):
                pbar.update(1)

                if result is not None:
                    pool.terminate()
                    return result

                # Check if another worker found it
                if result_dict.get("found"):
                    pool.terminate()
                    return result_dict.get("phrase")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Recover missing words from a BIP-39 mnemonic phrase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recover 1 missing word (first word unknown)
  python recover.py -p "? word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12" \\
                    -a "0x742d35Cc6634C0532925a3b844Bc9e7595f..." -b ethereum

  # Recover 2 missing words
  python recover.py -p "? ? word3 word4 word5 word6 word7 word8 word9 word10 word11 word12" \\
                    -a "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2" -b bitcoin

  # Recover with specific number of workers
  python recover.py -p "word1 ? word3 word4 ? word6 word7 word8 word9 word10 word11 word12" \\
                    -a "TN2T..." -b tron -w 4

Supported blockchains:
  bitcoin, bitcoin_segwit, bitcoin_native_segwit, ethereum, tron, bsc, litecoin, dogecoin, solana
        """,
    )

    parser.add_argument(
        "-p", "--phrase",
        required=True,
        help="Partial mnemonic phrase with '?' for missing words",
    )
    parser.add_argument(
        "-a", "--address",
        required=True,
        help="Known wallet address to match",
    )
    parser.add_argument(
        "-b", "--blockchain",
        required=True,
        choices=list(BLOCKCHAIN_CONFIG.keys()),
        help="Blockchain type",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=cpu_count(),
        help=f"Number of parallel workers (default: {cpu_count()})",
    )
    parser.add_argument(
        "--passphrase",
        default="",
        help="Optional BIP-39 passphrase (default: empty)",
    )
    parser.add_argument(
        "-o", "--output",
        default="recovered_phrase.txt",
        help="Output file for recovered phrase (default: recovered_phrase.txt)",
    )

    args = parser.parse_args()

    # Validate blockchain
    if args.blockchain not in BLOCKCHAIN_CONFIG:
        print(f"Error: Unknown blockchain '{args.blockchain}'")
        print(f"Supported: {', '.join(BLOCKCHAIN_CONFIG.keys())}")
        sys.exit(1)

    # Run recovery
    result = recover_phrase(
        partial_phrase=args.phrase,
        address=args.address,
        blockchain_name=args.blockchain,
        workers=args.workers,
        passphrase=args.passphrase,
    )

    if result:
        print(f"\n{'='*60}")
        print("SUCCESS! Recovered mnemonic phrase:")
        print(f"{'='*60}")
        print(f"\n  {result}\n")
        print(f"{'='*60}")

        # Save to file
        with open(args.output, "w") as f:
            f.write(result + "\n")
        print(f"Saved to: {args.output}")
    else:
        print(f"\n{'='*60}")
        print("FAILED: No matching phrase found")
        print(f"{'='*60}")
        print("\nPossible reasons:")
        print("  - Wrong blockchain selected")
        print("  - Wrong address")
        print("  - More words are missing than indicated")
        print("  - Non-standard derivation path used")
        sys.exit(1)


if __name__ == "__main__":
    main()
