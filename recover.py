#!/usr/bin/env python3
"""
BIP-39 Mnemonic Recovery Script

Recovers missing words from a partial BIP-39 mnemonic phrase by brute-forcing
possible combinations and checking against a known wallet address.

Supports: Bitcoin (BTC), Ethereum (ETH), TRON (TRX), Binance Smart Chain (BSC),
Litecoin (LTC), Dogecoin (DOGE), and more.
"""

import argparse
import itertools
import sys
from multiprocessing import Pool, cpu_count, Manager
from typing import Optional

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

# Global variables for multiprocessing
wordlist = None
target_address = None
blockchain = None
phrase_template = None
missing_indices = None
found_result = None
mnemonic_validator = None


def init_worker(wl, addr, bc, template, indices, result_dict):
    """Initialize worker process with shared data."""
    global wordlist, target_address, blockchain, phrase_template, missing_indices, found_result, mnemonic_validator
    wordlist = wl
    target_address = addr.lower() if addr.startswith("0x") else addr
    blockchain = bc
    phrase_template = template
    missing_indices = indices
    found_result = result_dict
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

    missing_count = len(indices)
    total_combinations = 2048 ** missing_count

    print(f"\n{'='*60}")
    print(f"BIP-39 Mnemonic Recovery")
    print(f"{'='*60}")
    print(f"Blockchain:       {blockchain_name}")
    print(f"Target Address:   {address}")
    print(f"Missing Words:    {missing_count} at position(s) {[i+1 for i in indices]}")
    print(f"Total Attempts:   {total_combinations:,}")
    print(f"Workers:          {workers}")
    print(f"{'='*60}\n")

    if missing_count > 3:
        print("WARNING: More than 3 missing words will take extremely long!")
        print(f"         Estimated combinations: {total_combinations:,}")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            return None

    # Generate all word combinations
    combinations = itertools.product(wl, repeat=missing_count)

    # Setup multiprocessing with shared state
    manager = Manager()
    result_dict = manager.dict()
    result_dict["found"] = False
    result_dict["phrase"] = None

    # Normalize target address
    addr_normalized = address.lower() if address.startswith("0x") else address

    # Process combinations in parallel with progress bar
    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(wl, addr_normalized, blockchain_name, template, indices, result_dict),
    ) as pool:
        with tqdm(total=total_combinations, desc="Checking", unit="combo") as pbar:
            # Use imap_unordered for better performance
            chunk_size = max(1, min(10000, total_combinations // (workers * 10)))

            for result in pool.imap_unordered(
                check_combination,
                itertools.product(wl, repeat=missing_count),
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
