#!/usr/bin/env python3
"""
BIP-39 Mnemonic Recovery Script

Recovers missing words from a partial BIP-39 mnemonic phrase by brute-forcing
possible combinations and checking against a known wallet address.

Supports: Bitcoin (BTC), Ethereum (ETH), TRON (TRX), Binance Smart Chain (BSC),
Litecoin (LTC), Dogecoin (DOGE), and more.

Features GPU acceleration (CUDA/OpenCL) with automatic CPU fallback.
Supports NVIDIA GPUs (CUDA) and Intel/AMD GPUs (OpenCL).
"""

import argparse
import hashlib
import itertools
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count, Manager
from typing import Optional, List, Tuple

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

# GPU backend detection
CUDA_AVAILABLE = False
OPENCL_AVAILABLE = False
GPU_AVAILABLE = False
GPU_NAME = "None"
GPU_BACKEND = None  # 'cuda', 'opencl', or None

# Try to import CUDA libraries (NVIDIA)
try:
    import numpy as np
    from numba import cuda
    import math

    # Check if CUDA is actually available
    if cuda.is_available():
        CUDA_AVAILABLE = True
        GPU_AVAILABLE = True
        GPU_NAME = cuda.get_current_device().name
        GPU_BACKEND = "cuda"
except ImportError:
    pass
except Exception:
    pass

# Try to import OpenCL libraries (Intel, AMD, and also NVIDIA)
try:
    import numpy as np
    import pyopencl as cl

    # Find available OpenCL platforms and devices
    OPENCL_PLATFORMS = []
    OPENCL_DEVICES = []

    for platform in cl.get_platforms():
        for device in platform.get_devices():
            device_type = device.type
            device_name = device.name.strip()
            vendor = device.vendor.strip()

            # Store device info
            OPENCL_DEVICES.append({
                "platform": platform,
                "device": device,
                "name": device_name,
                "vendor": vendor,
                "type": device_type,
            })

            # Prefer Intel/AMD GPU over CPU, but don't override CUDA if available
            if device_type == cl.device_type.GPU:
                if not CUDA_AVAILABLE:
                    OPENCL_AVAILABLE = True
                    GPU_AVAILABLE = True
                    GPU_NAME = device_name
                    GPU_BACKEND = "opencl"
                elif "Intel" in vendor or "AMD" in vendor:
                    # Also track Intel/AMD GPU even if CUDA is available
                    # User can explicitly choose OpenCL
                    OPENCL_AVAILABLE = True

    # If no GPU found but OpenCL CPU is available, note it (but don't use by default)
    if not GPU_AVAILABLE and OPENCL_DEVICES:
        OPENCL_AVAILABLE = True

except ImportError:
    pass
except Exception:
    pass

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

# SHA-256 constants for GPU kernel
SHA256_K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]

# Global variables for multiprocessing
wordlist = None
target_address = None
blockchain = None
phrase_template = None
missing_indices = None
found_result = None
mnemonic_validator = None
skip_checksum_validation = False

# State management for resumable sessions
DEFAULT_STATE_FILE = ".wallet_recovery_state.json"
SAVE_INTERVAL = 10000  # Save state every N combinations
_interrupt_requested = False


@dataclass
class RecoveryState:
    """Holds the state of a recovery session for persistence."""
    partial_phrase: str
    address: str
    blockchain: str
    passphrase: str
    device: str
    processed_count: int
    total_combinations: int
    session_id: str
    start_time: float
    elapsed_time: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RecoveryState":
        return cls(**data)


def generate_session_id(partial_phrase: str, address: str, blockchain: str) -> str:
    """Generate a unique session ID based on recovery parameters."""
    key = f"{partial_phrase}:{address}:{blockchain}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def save_state(state: RecoveryState, state_file: str = DEFAULT_STATE_FILE) -> None:
    """Save recovery state to file."""
    try:
        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
    except Exception as e:
        print(f"\nWarning: Failed to save state: {e}")


def load_state(state_file: str = DEFAULT_STATE_FILE) -> Optional[RecoveryState]:
    """Load recovery state from file."""
    try:
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                data = json.load(f)
                return RecoveryState.from_dict(data)
    except Exception as e:
        print(f"Warning: Failed to load state: {e}")
    return None


def clear_state(state_file: str = DEFAULT_STATE_FILE) -> None:
    """Clear the state file."""
    try:
        if os.path.exists(state_file):
            os.remove(state_file)
    except Exception:
        pass


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _interrupt_requested
    if _interrupt_requested:
        # Second interrupt - force exit
        print("\n\nForce exiting...")
        sys.exit(1)
    _interrupt_requested = True
    print("\n\nInterrupt received. Saving state and exiting gracefully...")
    print("(Press Ctrl+C again to force exit)")


def setup_signal_handlers():
    """Setup signal handlers for graceful interrupts."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def skip_combinations(generator, count: int):
    """Skip N combinations from the generator."""
    for _ in range(count):
        try:
            next(generator)
        except StopIteration:
            break


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


# ============================================================================
# GPU-accelerated functions (CUDA - NVIDIA)
# ============================================================================

if CUDA_AVAILABLE:
    # SHA-256 constants as numpy array for GPU
    SHA256_K_GPU = np.array(SHA256_K, dtype=np.uint32)

    @cuda.jit(device=True)
    def gpu_rotr(x, n):
        """Right rotate 32-bit integer."""
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    @cuda.jit(device=True)
    def gpu_sha256_transform(state, block, k):
        """SHA-256 compression function for a single 64-byte block."""
        # Initialize working variables
        a = state[0]
        b = state[1]
        c = state[2]
        d = state[3]
        e = state[4]
        f = state[5]
        g = state[6]
        h = state[7]

        # Prepare message schedule (w)
        w = cuda.local.array(64, dtype=np.uint32)
        for i in range(16):
            w[i] = block[i]

        for i in range(16, 64):
            s0 = gpu_rotr(w[i - 15], 7) ^ gpu_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
            s1 = gpu_rotr(w[i - 2], 17) ^ gpu_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & 0xFFFFFFFF

        # Main compression loop
        for i in range(64):
            S1 = gpu_rotr(e, 6) ^ gpu_rotr(e, 11) ^ gpu_rotr(e, 25)
            ch = (e & f) ^ ((~e) & g)
            temp1 = (h + S1 + ch + k[i] + w[i]) & 0xFFFFFFFF
            S0 = gpu_rotr(a, 2) ^ gpu_rotr(a, 13) ^ gpu_rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF

            h = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF

        # Add compressed chunk to current hash value
        state[0] = (state[0] + a) & 0xFFFFFFFF
        state[1] = (state[1] + b) & 0xFFFFFFFF
        state[2] = (state[2] + c) & 0xFFFFFFFF
        state[3] = (state[3] + d) & 0xFFFFFFFF
        state[4] = (state[4] + e) & 0xFFFFFFFF
        state[5] = (state[5] + f) & 0xFFFFFFFF
        state[6] = (state[6] + g) & 0xFFFFFFFF
        state[7] = (state[7] + h) & 0xFFFFFFFF

    @cuda.jit
    def gpu_validate_checksums_kernel(
        word_indices,  # Input: word indices for each combination [n_combos, n_missing]
        template_indices,  # Input: template with known word indices [phrase_len]
        missing_positions,  # Input: positions of missing words [n_missing]
        valid_flags,  # Output: 1 if valid checksum, 0 otherwise [n_combos]
        entropy_bits,
        checksum_bits,
        k_constants,  # SHA-256 K constants
    ):
        """
        GPU kernel to validate BIP-39 checksums in parallel.

        Each thread processes one combination.
        """
        idx = cuda.grid(1)
        if idx >= word_indices.shape[0]:
            return

        n_missing = missing_positions.shape[0]
        phrase_len = template_indices.shape[0]

        # Build complete phrase indices
        phrase_indices = cuda.local.array(24, dtype=np.uint32)  # Max 24 words
        for i in range(phrase_len):
            phrase_indices[i] = template_indices[i]

        # Fill in missing words from this combination
        for i in range(n_missing):
            pos = missing_positions[i]
            phrase_indices[pos] = word_indices[idx, i]

        # Convert word indices to entropy bits
        # Each word index is 11 bits
        total_bits = phrase_len * 11
        entropy_bytes_len = entropy_bits // 8

        # Create entropy byte array
        entropy = cuda.local.array(32, dtype=np.uint8)  # Max 256 bits = 32 bytes
        for i in range(32):
            entropy[i] = 0

        # Pack word indices into entropy bytes
        bit_pos = 0
        for i in range(phrase_len):
            word_idx = phrase_indices[i]
            # Place 11 bits of word_idx starting at bit_pos
            for b in range(11):
                if bit_pos < entropy_bits:  # Only entropy bits, not checksum
                    bit_val = (word_idx >> (10 - b)) & 1
                    byte_idx = bit_pos // 8
                    bit_offset = 7 - (bit_pos % 8)
                    entropy[byte_idx] |= (bit_val << bit_offset)
                bit_pos += 1

        # Compute SHA-256 of entropy
        # Initialize state
        state = cuda.local.array(8, dtype=np.uint32)
        state[0] = 0x6A09E667
        state[1] = 0xBB67AE85
        state[2] = 0x3C6EF372
        state[3] = 0xA54FF53A
        state[4] = 0x510E527F
        state[5] = 0x9B05688C
        state[6] = 0x1F83D9AB
        state[7] = 0x5BE0CD19

        # Prepare padded message block (single block for <= 55 bytes)
        block = cuda.local.array(16, dtype=np.uint32)
        for i in range(16):
            block[i] = 0

        # Copy entropy bytes to block (big-endian)
        for i in range(entropy_bytes_len):
            word_idx = i // 4
            byte_offset = 3 - (i % 4)
            block[word_idx] |= entropy[i] << (byte_offset * 8)

        # Add padding bit
        pad_byte_idx = entropy_bytes_len
        pad_word_idx = pad_byte_idx // 4
        pad_byte_offset = 3 - (pad_byte_idx % 4)
        block[pad_word_idx] |= 0x80 << (pad_byte_offset * 8)

        # Add length in bits at the end (last 64 bits)
        block[15] = entropy_bits

        # Transform
        gpu_sha256_transform(state, block, k_constants)

        # Extract first byte of hash for checksum
        hash_first_byte = (state[0] >> 24) & 0xFF
        computed_checksum = hash_first_byte >> (8 - checksum_bits)

        # Extract checksum from mnemonic (last checksum_bits of last word)
        last_word_idx = phrase_indices[phrase_len - 1]
        mnemonic_checksum = last_word_idx & ((1 << checksum_bits) - 1)

        # Compare
        if computed_checksum == mnemonic_checksum:
            valid_flags[idx] = 1
        else:
            valid_flags[idx] = 0


def recover_phrase_cuda(
    partial_phrase: str,
    address: str,
    blockchain_name: str,
    workers: int = None,
    passphrase: str = "",
    start_offset: int = 0,
    state_file: str = DEFAULT_STATE_FILE,
    state: RecoveryState = None,
) -> Optional[str]:
    """
    CUDA GPU-accelerated recovery of missing words from a partial mnemonic.

    Uses NVIDIA GPU for batch checksum validation, then CPU for address derivation.

    Args:
        partial_phrase: Mnemonic with '?' for missing words
        address: Target wallet address
        blockchain_name: Blockchain name
        workers: Number of CPU workers for address derivation
        passphrase: Optional BIP-39 passphrase
        start_offset: Number of combinations to skip (for resuming)
        state_file: Path to state file for persistence
        state: Existing state object (for resuming)

    Returns:
        Recovered mnemonic or None
    """
    global _interrupt_requested

    if workers is None:
        workers = cpu_count()

    # Load BIP-39 wordlist
    mnemo = Mnemonic("english")
    wl = mnemo.wordlist
    word_to_idx = {word: idx for idx, word in enumerate(wl)}

    # Parse phrase template
    words = partial_phrase.strip().split()
    template = words.copy()
    indices = [i for i, w in enumerate(words) if w == "?"]

    if not indices:
        print("No missing words indicated (use '?' as placeholder)")
        return None

    phrase_len = len(words)
    missing_count = len(indices)

    if phrase_len not in PHRASE_CONFIG:
        print(f"Error: Invalid phrase length {phrase_len}")
        return None

    entropy_bits, checksum_bits = PHRASE_CONFIG[phrase_len]

    # Build template indices (use 0 as placeholder for missing)
    template_indices = np.array(
        [word_to_idx.get(w, 0) for w in words], dtype=np.uint32
    )
    missing_positions = np.array(indices, dtype=np.uint32)

    # Determine word candidates
    word_candidates = []
    optimization_applied = False

    for idx in indices:
        if idx == phrase_len - 1 and missing_count == 1:
            known_words = [w for w in words if w != "?"]
            valid_words = compute_valid_last_words(known_words, wl)
            word_candidates.append([word_to_idx[w] for w in valid_words])
            optimization_applied = True
        else:
            word_candidates.append(list(range(2048)))

    total_combinations = 1
    for candidates in word_candidates:
        total_combinations *= len(candidates)

    print(f"\n{'='*60}")
    print(f"BIP-39 Mnemonic Recovery (CUDA Mode)")
    print(f"{'='*60}")
    print(f"GPU Device:       {GPU_NAME}")
    print(f"Blockchain:       {blockchain_name}")
    print(f"Target Address:   {address}")
    print(f"Phrase Length:    {phrase_len} words")
    print(f"Missing Words:    {missing_count} at position(s) {[i+1 for i in indices]}")
    if optimization_applied:
        print(f"Optimization:     Last-word entropy constraint applied")
        print(f"Valid Candidates: {len(word_candidates[0])} (reduced from 2048)")
    print(f"Total Attempts:   {total_combinations:,}")
    if start_offset > 0:
        print(f"Resuming from:    {start_offset:,} ({100*start_offset/total_combinations:.1f}%)")
    print(f"CPU Workers:      {workers}")
    print(f"{'='*60}\n")

    if missing_count > 3 and start_offset == 0:
        print("WARNING: More than 3 missing words will take extremely long!")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            return None

    # Normalize target address
    addr_normalized = address.lower() if address.startswith("0x") else address

    # GPU batch processing parameters
    batch_size = min(1_000_000, total_combinations)  # Process up to 1M at a time

    # Copy constants to GPU
    k_gpu = cuda.to_device(SHA256_K_GPU)
    template_gpu = cuda.to_device(template_indices)
    positions_gpu = cuda.to_device(missing_positions)

    # Generate combinations in batches
    combo_generator = itertools.product(*word_candidates)

    # Skip already processed combinations if resuming
    if start_offset > 0:
        print(f"Skipping {start_offset:,} already processed combinations...")
        skip_combinations(combo_generator, start_offset)

    # Setup for address verification
    validator = Bip39MnemonicValidator(Bip39Languages.ENGLISH)

    # Initialize or update state
    if state is None:
        session_id = generate_session_id(partial_phrase, address, blockchain_name)
        state = RecoveryState(
            partial_phrase=partial_phrase,
            address=address,
            blockchain=blockchain_name,
            passphrase=passphrase,
            device="cuda",
            processed_count=start_offset,
            total_combinations=total_combinations,
            session_id=session_id,
            start_time=time.time(),
        )

    last_save_count = start_offset

    with tqdm(total=total_combinations, desc="Checking (CUDA)", unit="combo", initial=start_offset) as pbar:
        processed = start_offset

        while processed < total_combinations and not _interrupt_requested:
            # Collect batch of combinations
            batch_combos = []
            for _ in range(min(batch_size, total_combinations - processed)):
                try:
                    combo = next(combo_generator)
                    batch_combos.append(combo)
                except StopIteration:
                    break

            if not batch_combos:
                break

            current_batch_size = len(batch_combos)

            # Convert to numpy array
            word_indices = np.array(batch_combos, dtype=np.uint32)

            # Allocate output array
            valid_flags = np.zeros(current_batch_size, dtype=np.uint32)

            # Copy to GPU
            word_indices_gpu = cuda.to_device(word_indices)
            valid_flags_gpu = cuda.to_device(valid_flags)

            # Launch kernel
            threads_per_block = 256
            blocks = (current_batch_size + threads_per_block - 1) // threads_per_block

            gpu_validate_checksums_kernel[blocks, threads_per_block](
                word_indices_gpu,
                template_gpu,
                positions_gpu,
                valid_flags_gpu,
                entropy_bits,
                checksum_bits,
                k_gpu,
            )

            # Copy results back
            valid_flags = valid_flags_gpu.copy_to_host()

            # Get valid combinations
            valid_indices = np.where(valid_flags == 1)[0]

            # Check valid combinations on CPU
            for vi in valid_indices:
                combo = batch_combos[vi]
                # Build phrase
                phrase = template.copy()
                for pos_idx, word_idx in zip(indices, combo):
                    phrase[pos_idx] = wl[word_idx]

                mnemonic_str = " ".join(phrase)

                # Double-check validity (should always pass)
                try:
                    if not validator.IsValid(mnemonic_str):
                        continue
                except Exception:
                    continue

                # Derive address
                derived = derive_address(mnemonic_str, blockchain_name)
                if derived is None:
                    continue

                derived_normalized = derived.lower() if derived.startswith("0x") else derived

                if derived_normalized == addr_normalized:
                    # Clear state on success
                    clear_state(state_file)
                    return mnemonic_str

            processed += current_batch_size
            pbar.update(current_batch_size)

            # Periodically save state
            if processed - last_save_count >= SAVE_INTERVAL:
                state.processed_count = processed
                state.elapsed_time = time.time() - state.start_time
                save_state(state, state_file)
                last_save_count = processed

    # Save final state if interrupted
    if _interrupt_requested:
        state.processed_count = processed
        state.elapsed_time = time.time() - state.start_time
        save_state(state, state_file)
        print(f"\nState saved. Processed {processed:,} of {total_combinations:,} combinations.")
        print(f"Run with --resume to continue from where you left off.")
        return None

    return None


# ============================================================================
# GPU-accelerated functions (OpenCL - Intel, AMD, NVIDIA)
# ============================================================================

# OpenCL kernel for SHA-256 checksum validation
OPENCL_KERNEL_SOURCE = """
__constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint rotr(uint x, uint n) {
    return (x >> n) | (x << (32 - n));
}

__kernel void validate_checksums(
    __global const uint *word_indices,      // [n_combos * n_missing]
    __global const uint *template_indices,  // [phrase_len]
    __global const uint *missing_positions, // [n_missing]
    __global uint *valid_flags,             // [n_combos]
    const uint n_missing,
    const uint phrase_len,
    const uint entropy_bits,
    const uint checksum_bits
) {
    uint idx = get_global_id(0);

    // Build complete phrase indices
    uint phrase_indices[24];  // Max 24 words
    for (uint i = 0; i < phrase_len; i++) {
        phrase_indices[i] = template_indices[i];
    }

    // Fill in missing words from this combination
    for (uint i = 0; i < n_missing; i++) {
        uint pos = missing_positions[i];
        phrase_indices[pos] = word_indices[idx * n_missing + i];
    }

    // Create entropy byte array
    uchar entropy[32];
    for (uint i = 0; i < 32; i++) {
        entropy[i] = 0;
    }

    // Pack word indices into entropy bytes (11 bits per word)
    uint bit_pos = 0;
    for (uint i = 0; i < phrase_len; i++) {
        uint word_idx = phrase_indices[i];
        for (uint b = 0; b < 11; b++) {
            if (bit_pos < entropy_bits) {
                uint bit_val = (word_idx >> (10 - b)) & 1;
                uint byte_idx = bit_pos / 8;
                uint bit_offset = 7 - (bit_pos % 8);
                entropy[byte_idx] |= (bit_val << bit_offset);
            }
            bit_pos++;
        }
    }

    uint entropy_bytes_len = entropy_bits / 8;

    // SHA-256 initialization
    uint state[8];
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;

    // Prepare padded message block
    uint block[16];
    for (uint i = 0; i < 16; i++) {
        block[i] = 0;
    }

    // Copy entropy bytes to block (big-endian)
    for (uint i = 0; i < entropy_bytes_len; i++) {
        uint word_idx_blk = i / 4;
        uint byte_offset = 3 - (i % 4);
        block[word_idx_blk] |= ((uint)entropy[i]) << (byte_offset * 8);
    }

    // Add padding bit
    uint pad_byte_idx = entropy_bytes_len;
    uint pad_word_idx = pad_byte_idx / 4;
    uint pad_byte_offset = 3 - (pad_byte_idx % 4);
    block[pad_word_idx] |= 0x80 << (pad_byte_offset * 8);

    // Add length in bits
    block[15] = entropy_bits;

    // SHA-256 compression
    uint w[64];
    for (uint i = 0; i < 16; i++) {
        w[i] = block[i];
    }
    for (uint i = 16; i < 64; i++) {
        uint s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];

    for (uint i = 0; i < 64; i++) {
        uint S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint ch = (e & f) ^ ((~e) & g);
        uint temp1 = h + S1 + ch + K[i] + w[i];
        uint S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a;

    // Extract checksum from hash
    uint hash_first_byte = (state[0] >> 24) & 0xFF;
    uint computed_checksum = hash_first_byte >> (8 - checksum_bits);

    // Extract checksum from mnemonic
    uint last_word_idx = phrase_indices[phrase_len - 1];
    uint mnemonic_checksum = last_word_idx & ((1 << checksum_bits) - 1);

    valid_flags[idx] = (computed_checksum == mnemonic_checksum) ? 1 : 0;
}
"""


def get_opencl_device(prefer_gpu: bool = True):
    """
    Get the best available OpenCL device.

    Args:
        prefer_gpu: If True, prefer GPU over CPU

    Returns:
        Tuple of (context, queue, device_name) or (None, None, None)
    """
    if not OPENCL_AVAILABLE:
        return None, None, None

    try:
        import pyopencl as cl

        best_device = None
        best_device_info = None

        for device_info in OPENCL_DEVICES:
            device = device_info["device"]
            is_gpu = device_info["type"] == cl.device_type.GPU

            if prefer_gpu:
                if is_gpu:
                    # Prefer Intel/AMD GPU over others for OpenCL
                    if best_device is None or "Intel" in device_info["vendor"] or "AMD" in device_info["vendor"]:
                        best_device = device
                        best_device_info = device_info
            else:
                # Just take any device
                if best_device is None:
                    best_device = device
                    best_device_info = device_info

        if best_device is None and OPENCL_DEVICES:
            # Fallback to first available device
            best_device = OPENCL_DEVICES[0]["device"]
            best_device_info = OPENCL_DEVICES[0]

        if best_device:
            ctx = cl.Context([best_device])
            queue = cl.CommandQueue(ctx)
            return ctx, queue, best_device_info["name"]

    except Exception:
        pass

    return None, None, None


def recover_phrase_opencl(
    partial_phrase: str,
    address: str,
    blockchain_name: str,
    workers: int = None,
    passphrase: str = "",
    start_offset: int = 0,
    state_file: str = DEFAULT_STATE_FILE,
    state: RecoveryState = None,
) -> Optional[str]:
    """
    OpenCL GPU-accelerated recovery of missing words from a partial mnemonic.

    Uses OpenCL (Intel/AMD/NVIDIA GPU) for batch checksum validation,
    then CPU for address derivation.

    Args:
        partial_phrase: Mnemonic with '?' for missing words
        address: Target wallet address
        blockchain_name: Blockchain name
        workers: Number of CPU workers for address derivation
        passphrase: Optional BIP-39 passphrase
        start_offset: Number of combinations to skip (for resuming)
        state_file: Path to state file for persistence
        state: Existing state object (for resuming)

    Returns:
        Recovered mnemonic or None
    """
    global _interrupt_requested

    import pyopencl as cl

    if workers is None:
        workers = cpu_count()

    # Get OpenCL context and queue
    ctx, queue, device_name = get_opencl_device(prefer_gpu=True)
    if ctx is None:
        print("ERROR: No OpenCL device available")
        return None

    # Load BIP-39 wordlist
    mnemo = Mnemonic("english")
    wl = mnemo.wordlist
    word_to_idx = {word: idx for idx, word in enumerate(wl)}

    # Parse phrase template
    words = partial_phrase.strip().split()
    template = words.copy()
    indices = [i for i, w in enumerate(words) if w == "?"]

    if not indices:
        print("No missing words indicated (use '?' as placeholder)")
        return None

    phrase_len = len(words)
    missing_count = len(indices)

    if phrase_len not in PHRASE_CONFIG:
        print(f"Error: Invalid phrase length {phrase_len}")
        return None

    entropy_bits, checksum_bits = PHRASE_CONFIG[phrase_len]

    # Build template indices
    template_indices = np.array(
        [word_to_idx.get(w, 0) for w in words], dtype=np.uint32
    )
    missing_positions = np.array(indices, dtype=np.uint32)

    # Determine word candidates
    word_candidates = []
    optimization_applied = False

    for idx in indices:
        if idx == phrase_len - 1 and missing_count == 1:
            known_words = [w for w in words if w != "?"]
            valid_words = compute_valid_last_words(known_words, wl)
            word_candidates.append([word_to_idx[w] for w in valid_words])
            optimization_applied = True
        else:
            word_candidates.append(list(range(2048)))

    total_combinations = 1
    for candidates in word_candidates:
        total_combinations *= len(candidates)

    print(f"\n{'='*60}")
    print(f"BIP-39 Mnemonic Recovery (OpenCL Mode)")
    print(f"{'='*60}")
    print(f"GPU Device:       {device_name}")
    print(f"Blockchain:       {blockchain_name}")
    print(f"Target Address:   {address}")
    print(f"Phrase Length:    {phrase_len} words")
    print(f"Missing Words:    {missing_count} at position(s) {[i+1 for i in indices]}")
    if optimization_applied:
        print(f"Optimization:     Last-word entropy constraint applied")
        print(f"Valid Candidates: {len(word_candidates[0])} (reduced from 2048)")
    print(f"Total Attempts:   {total_combinations:,}")
    if start_offset > 0:
        print(f"Resuming from:    {start_offset:,} ({100*start_offset/total_combinations:.1f}%)")
    print(f"CPU Workers:      {workers}")
    print(f"{'='*60}\n")

    if missing_count > 3 and start_offset == 0:
        print("WARNING: More than 3 missing words will take extremely long!")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            return None

    # Normalize target address
    addr_normalized = address.lower() if address.startswith("0x") else address

    # Compile OpenCL kernel
    try:
        prg = cl.Program(ctx, OPENCL_KERNEL_SOURCE).build()
        # Retrieve kernel once to avoid repeated retrieval overhead
        kernel = cl.Kernel(prg, "validate_checksums")
    except cl.RuntimeError as e:
        print(f"ERROR: Failed to compile OpenCL kernel: {e}")
        return None

    # GPU batch processing parameters
    batch_size = min(500_000, total_combinations)  # Process up to 500K at a time

    # Create OpenCL buffers for template and positions (these don't change)
    mf = cl.mem_flags
    template_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=template_indices)
    positions_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=missing_positions)

    # Generate combinations in batches
    combo_generator = itertools.product(*word_candidates)

    # Skip already processed combinations if resuming
    if start_offset > 0:
        print(f"Skipping {start_offset:,} already processed combinations...")
        skip_combinations(combo_generator, start_offset)

    # Setup for address verification
    validator = Bip39MnemonicValidator(Bip39Languages.ENGLISH)

    # Initialize or update state
    if state is None:
        session_id = generate_session_id(partial_phrase, address, blockchain_name)
        state = RecoveryState(
            partial_phrase=partial_phrase,
            address=address,
            blockchain=blockchain_name,
            passphrase=passphrase,
            device="opencl",
            processed_count=start_offset,
            total_combinations=total_combinations,
            session_id=session_id,
            start_time=time.time(),
        )

    last_save_count = start_offset

    with tqdm(total=total_combinations, desc="Checking (OpenCL)", unit="combo", initial=start_offset) as pbar:
        processed = start_offset

        while processed < total_combinations and not _interrupt_requested:
            # Collect batch of combinations
            batch_combos = []
            for _ in range(min(batch_size, total_combinations - processed)):
                try:
                    combo = next(combo_generator)
                    batch_combos.append(combo)
                except StopIteration:
                    break

            if not batch_combos:
                break

            current_batch_size = len(batch_combos)

            # Flatten word indices for OpenCL
            word_indices = np.array(batch_combos, dtype=np.uint32).flatten()
            valid_flags = np.zeros(current_batch_size, dtype=np.uint32)

            # Create OpenCL buffers
            word_indices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=word_indices)
            valid_flags_buf = cl.Buffer(ctx, mf.WRITE_ONLY, valid_flags.nbytes)

            # Execute kernel (using pre-retrieved kernel instance)
            kernel(
                queue,
                (current_batch_size,),
                None,  # Let OpenCL choose work group size
                word_indices_buf,
                template_buf,
                positions_buf,
                valid_flags_buf,
                np.uint32(missing_count),
                np.uint32(phrase_len),
                np.uint32(entropy_bits),
                np.uint32(checksum_bits),
            )

            # Read results back
            cl.enqueue_copy(queue, valid_flags, valid_flags_buf)
            queue.finish()

            # Get valid combinations
            valid_indices = np.where(valid_flags == 1)[0]

            # Check valid combinations on CPU
            for vi in valid_indices:
                combo = batch_combos[vi]
                # Build phrase
                phrase = template.copy()
                for pos_idx, word_idx in zip(indices, combo):
                    phrase[pos_idx] = wl[word_idx]

                mnemonic_str = " ".join(phrase)

                # Double-check validity
                try:
                    if not validator.IsValid(mnemonic_str):
                        continue
                except Exception:
                    continue

                # Derive address
                derived = derive_address(mnemonic_str, blockchain_name)
                if derived is None:
                    continue

                derived_normalized = derived.lower() if derived.startswith("0x") else derived

                if derived_normalized == addr_normalized:
                    # Clear state on success
                    clear_state(state_file)
                    return mnemonic_str

            processed += current_batch_size
            pbar.update(current_batch_size)

            # Periodically save state
            if processed - last_save_count >= SAVE_INTERVAL:
                state.processed_count = processed
                state.elapsed_time = time.time() - state.start_time
                save_state(state, state_file)
                last_save_count = processed

    # Save final state if interrupted
    if _interrupt_requested:
        state.processed_count = processed
        state.elapsed_time = time.time() - state.start_time
        save_state(state, state_file)
        print(f"\nState saved. Processed {processed:,} of {total_combinations:,} combinations.")
        print(f"Run with --resume to continue from where you left off.")
        return None

    return None


def recover_phrase_cpu(
    partial_phrase: str,
    address: str,
    blockchain_name: str,
    workers: int = None,
    passphrase: str = "",
    start_offset: int = 0,
    state_file: str = DEFAULT_STATE_FILE,
    state: RecoveryState = None,
) -> Optional[str]:
    """
    CPU-based recovery of missing words from a partial mnemonic.

    Uses multiprocessing for parallel processing.

    Args:
        partial_phrase: Mnemonic with '?' for missing words
        address: Target wallet address
        blockchain_name: Blockchain name
        workers: Number of parallel workers (default: CPU count)
        passphrase: Optional BIP-39 passphrase
        start_offset: Number of combinations to skip (for resuming)
        state_file: Path to state file for persistence
        state: Existing state object (for resuming)

    Returns:
        Recovered mnemonic or None
    """
    global _interrupt_requested

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
    word_candidates = []
    optimization_applied = False

    for idx in indices:
        if idx == phrase_len - 1 and missing_count == 1:
            known_words = [w for w in words if w != "?"]
            valid_words = compute_valid_last_words(known_words, wl)
            word_candidates.append(valid_words)
            optimization_applied = True
        elif idx == phrase_len - 1:
            word_candidates.append(wl)
        else:
            word_candidates.append(wl)

    total_combinations = 1
    for candidates in word_candidates:
        total_combinations *= len(candidates)

    print(f"\n{'='*60}")
    print(f"BIP-39 Mnemonic Recovery (CPU Mode)")
    print(f"{'='*60}")
    print(f"Blockchain:       {blockchain_name}")
    print(f"Target Address:   {address}")
    print(f"Phrase Length:    {phrase_len} words")
    print(f"Missing Words:    {missing_count} at position(s) {[i+1 for i in indices]}")
    if optimization_applied:
        print(f"Optimization:     Last-word entropy constraint applied")
        print(f"Valid Candidates: {len(word_candidates[0])} (reduced from 2048)")
    print(f"Total Attempts:   {total_combinations:,}")
    if start_offset > 0:
        print(f"Resuming from:    {start_offset:,} ({100*start_offset/total_combinations:.1f}%)")
    print(f"Workers:          {workers}")
    print(f"{'='*60}\n")

    if missing_count > 3 and start_offset == 0:
        print("WARNING: More than 3 missing words will take extremely long!")
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            return None

    # Setup multiprocessing with shared state
    manager = Manager()
    result_dict = manager.dict()
    result_dict["found"] = False
    result_dict["phrase"] = None

    # Normalize target address
    addr_normalized = address.lower() if address.startswith("0x") else address

    # Create combination generator and skip already processed
    combo_generator = itertools.product(*word_candidates)
    if start_offset > 0:
        print(f"Skipping {start_offset:,} already processed combinations...")
        skip_combinations(combo_generator, start_offset)

    # Initialize or update state
    if state is None:
        session_id = generate_session_id(partial_phrase, address, blockchain_name)
        state = RecoveryState(
            partial_phrase=partial_phrase,
            address=address,
            blockchain=blockchain_name,
            passphrase=passphrase,
            device="cpu",
            processed_count=start_offset,
            total_combinations=total_combinations,
            session_id=session_id,
            start_time=time.time(),
        )

    processed = start_offset
    last_save_count = start_offset

    # Process combinations in parallel with progress bar
    try:
        with Pool(
            processes=workers,
            initializer=init_worker,
            initargs=(wl, addr_normalized, blockchain_name, template, indices, result_dict, optimization_applied),
        ) as pool:
            with tqdm(total=total_combinations, desc="Checking (CPU)", unit="combo", initial=start_offset) as pbar:
                chunk_size = max(1, min(10000, total_combinations // (workers * 10)))

                for result in pool.imap_unordered(
                    check_combination,
                    combo_generator,
                    chunksize=chunk_size,
                ):
                    processed += 1
                    pbar.update(1)

                    # Periodically save state
                    if processed - last_save_count >= SAVE_INTERVAL:
                        state.processed_count = processed
                        state.elapsed_time = time.time() - state.start_time
                        save_state(state, state_file)
                        last_save_count = processed

                    # Check for interrupt
                    if _interrupt_requested:
                        pool.terminate()
                        break

                    if result is not None:
                        pool.terminate()
                        clear_state(state_file)
                        return result

                    if result_dict.get("found"):
                        pool.terminate()
                        clear_state(state_file)
                        return result_dict.get("phrase")

    except KeyboardInterrupt:
        pass  # Signal handler will set _interrupt_requested

    # Save final state if interrupted
    if _interrupt_requested:
        state.processed_count = processed
        state.elapsed_time = time.time() - state.start_time
        save_state(state, state_file)
        print(f"\nState saved. Processed {processed:,} of {total_combinations:,} combinations.")
        print(f"Run with --resume to continue from where you left off.")
        return None

    return None


def recover_phrase(
    partial_phrase: str,
    address: str,
    blockchain_name: str,
    workers: int = None,
    passphrase: str = "",
    device: str = "auto",
    start_offset: int = 0,
    state_file: str = DEFAULT_STATE_FILE,
    state: RecoveryState = None,
) -> Optional[str]:
    """
    Attempt to recover missing words from a partial mnemonic.

    Automatically selects GPU or CPU based on availability and device parameter.

    Args:
        partial_phrase: Mnemonic with '?' for missing words
        address: Target wallet address
        blockchain_name: Blockchain name
        workers: Number of parallel workers (default: CPU count)
        passphrase: Optional BIP-39 passphrase
        device: Device to use ('auto', 'cpu', 'gpu', 'cuda', 'opencl')
        start_offset: Number of combinations to skip (for resuming)
        state_file: Path to state file for persistence
        state: Existing state object (for resuming)

    Returns:
        Recovered mnemonic or None
    """
    # Determine which backend to use
    backend = None  # 'cuda', 'opencl', or None (CPU)

    if device == "cuda":
        if CUDA_AVAILABLE:
            backend = "cuda"
        else:
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
    elif device == "opencl":
        if OPENCL_AVAILABLE:
            backend = "opencl"
        else:
            print("WARNING: OpenCL requested but not available. Falling back to CPU.")
    elif device == "gpu":
        # Generic GPU request - prefer CUDA, then OpenCL
        if CUDA_AVAILABLE:
            backend = "cuda"
        elif OPENCL_AVAILABLE:
            backend = "opencl"
        else:
            print("WARNING: GPU requested but not available. Falling back to CPU.")
    elif device == "auto":
        # Auto-detect best available backend
        if CUDA_AVAILABLE:
            backend = "cuda"
        elif OPENCL_AVAILABLE:
            backend = "opencl"
        # else: backend remains None (CPU)

    # Execute with selected backend
    if backend == "cuda":
        return recover_phrase_cuda(
            partial_phrase, address, blockchain_name, workers, passphrase,
            start_offset, state_file, state
        )
    elif backend == "opencl":
        return recover_phrase_opencl(
            partial_phrase, address, blockchain_name, workers, passphrase,
            start_offset, state_file, state
        )
    else:
        return recover_phrase_cpu(
            partial_phrase, address, blockchain_name, workers, passphrase,
            start_offset, state_file, state
        )


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

  # Force GPU mode (auto-select best GPU backend)
  python recover.py -p "? word2 ..." -a "0x..." -b ethereum --device gpu

  # Force CUDA (NVIDIA GPU)
  python recover.py -p "? word2 ..." -a "0x..." -b ethereum --device cuda

  # Force OpenCL (Intel/AMD GPU)
  python recover.py -p "? word2 ..." -a "0x..." -b ethereum --device opencl

  # Force CPU mode
  python recover.py -p "? word2 ..." -a "0x..." -b ethereum --device cpu

  # Resume an interrupted session
  python recover.py --resume

  # Resume from a custom state file
  python recover.py --resume --state-file my_state.json

  # Clear saved state and start fresh
  python recover.py -p "..." -a "..." -b ethereum --clear-state

Supported blockchains:
  bitcoin, bitcoin_segwit, bitcoin_native_segwit, ethereum, tron, bsc, litecoin, dogecoin, solana

Supported GPU backends:
  - CUDA: NVIDIA GPUs (requires numba + CUDA toolkit)
  - OpenCL: Intel/AMD/NVIDIA GPUs (requires pyopencl + OpenCL runtime)

State Management:
  The script automatically saves progress periodically. If interrupted (Ctrl+C),
  you can resume from where you left off using --resume. State is saved to
  .wallet_recovery_state.json by default.
        """,
    )

    parser.add_argument(
        "-p", "--phrase",
        help="Partial mnemonic phrase with '?' for missing words",
    )
    parser.add_argument(
        "-a", "--address",
        help="Known wallet address to match",
    )
    parser.add_argument(
        "-b", "--blockchain",
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
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cpu", "gpu", "cuda", "opencl"],
        default="auto",
        help="Device to use for computation: auto (default), cpu, gpu (best available), cuda (NVIDIA), opencl (Intel/AMD)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previously interrupted session",
    )
    parser.add_argument(
        "--state-file",
        default=DEFAULT_STATE_FILE,
        help=f"State file for saving/resuming progress (default: {DEFAULT_STATE_FILE})",
    )
    parser.add_argument(
        "--clear-state",
        action="store_true",
        help="Clear any saved state and start fresh",
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Show saved state information and exit",
    )

    args = parser.parse_args()

    # Handle --show-state
    if args.show_state:
        state = load_state(args.state_file)
        if state:
            elapsed = state.elapsed_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"\nSaved Recovery State ({args.state_file}):")
            print(f"{'='*60}")
            print(f"Session ID:       {state.session_id}")
            print(f"Phrase:           {state.partial_phrase}")
            print(f"Address:          {state.address}")
            print(f"Blockchain:       {state.blockchain}")
            print(f"Device:           {state.device}")
            print(f"Progress:         {state.processed_count:,} / {state.total_combinations:,}")
            print(f"Percentage:       {100*state.processed_count/state.total_combinations:.2f}%")
            print(f"Elapsed Time:     {hours}h {minutes}m {seconds}s")
            print(f"{'='*60}")
            print("\nUse --resume to continue this session.")
        else:
            print(f"No saved state found at {args.state_file}")
        sys.exit(0)

    # Handle --clear-state
    if args.clear_state:
        clear_state(args.state_file)
        print(f"Cleared saved state from {args.state_file}")

    # Handle --resume
    state = None
    start_offset = 0
    if args.resume:
        state = load_state(args.state_file)
        if state:
            print(f"\nResuming previous session (ID: {state.session_id})")
            print(f"Progress: {state.processed_count:,} / {state.total_combinations:,} ({100*state.processed_count/state.total_combinations:.1f}%)")
            # Use parameters from saved state
            args.phrase = state.partial_phrase
            args.address = state.address
            args.blockchain = state.blockchain
            args.passphrase = state.passphrase
            args.device = state.device
            start_offset = state.processed_count
        else:
            print(f"No saved state found at {args.state_file}")
            print("Cannot resume. Please start a new session with -p, -a, -b options.")
            sys.exit(1)

    # Validate required arguments if not resuming
    if not args.resume or state is None:
        if not args.phrase:
            parser.error("the following arguments are required: -p/--phrase")
        if not args.address:
            parser.error("the following arguments are required: -a/--address")
        if not args.blockchain:
            parser.error("the following arguments are required: -b/--blockchain")

    # Setup signal handlers for graceful interrupts
    setup_signal_handlers()

    # Show device info
    print("GPU Backend Detection:")
    if CUDA_AVAILABLE:
        print(f"  - CUDA:   Available ({GPU_NAME if GPU_BACKEND == 'cuda' else 'NVIDIA GPU'})")
    else:
        print("  - CUDA:   Not available")
    if OPENCL_AVAILABLE:
        opencl_gpu_names = [d["name"] for d in OPENCL_DEVICES if d["type"] == 4]  # cl.device_type.GPU == 4
        if opencl_gpu_names:
            print(f"  - OpenCL: Available ({', '.join(opencl_gpu_names)})")
        else:
            print("  - OpenCL: Available (CPU only)")
    else:
        print("  - OpenCL: Not available")

    if args.device == "auto":
        if CUDA_AVAILABLE:
            print(f"Selected: CUDA ({GPU_NAME})")
        elif OPENCL_AVAILABLE:
            print(f"Selected: OpenCL ({GPU_NAME})")
        else:
            print("Selected: CPU (no GPU available)")

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
        device=args.device,
        start_offset=start_offset,
        state_file=args.state_file,
        state=state,
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

        # Clear state on success
        clear_state(args.state_file)
    elif _interrupt_requested:
        # Interrupted - state already saved
        print("\nSession interrupted. State has been saved.")
        print(f"Use 'python recover.py --resume' to continue.")
        sys.exit(130)  # Standard exit code for SIGINT
    else:
        print(f"\n{'='*60}")
        print("FAILED: No matching phrase found")
        print(f"{'='*60}")
        print("\nPossible reasons:")
        print("  - Wrong blockchain selected")
        print("  - Wrong address")
        print("  - More words are missing than indicated")
        print("  - Non-standard derivation path used")

        # Clear state since we exhausted all combinations
        clear_state(args.state_file)
        sys.exit(1)


if __name__ == "__main__":
    main()
