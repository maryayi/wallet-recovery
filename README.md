# Wallet Recovery

A Python tool to recover missing words from BIP-39 mnemonic seed phrases by brute-forcing possible combinations and validating against a known wallet address.

## Features

- **Multi-Blockchain Support**: Bitcoin, Ethereum, TRON, BSC, Litecoin, Dogecoin, Solana
- **Multiple Bitcoin Address Formats**: Legacy, SegWit, Native SegWit
- **GPU Acceleration**: CUDA support for massively parallel checksum validation (auto-detects GPU)
- **CPU Fallback**: Automatically falls back to multi-core CPU if no GPU available
- **Fast Validation**: Checksum validation before address derivation for quick rejection
- **Parallel Processing**: Multi-core CPU support for faster recovery
- **Progress Tracking**: Real-time progress bar with completion estimates
- **Flexible Input**: Mark unknown words with `?` placeholder

## Supported Blockchains

| Blockchain              | Flag                    | Address Prefix | Derivation Path   |
| ----------------------- | ----------------------- | -------------- | ----------------- |
| Bitcoin (Legacy)        | `bitcoin`               | `1...`         | m/44'/0'/0'/0/0   |
| Bitcoin (SegWit)        | `bitcoin_segwit`        | `3...`         | m/49'/0'/0'/0/0   |
| Bitcoin (Native SegWit) | `bitcoin_native_segwit` | `bc1...`       | m/84'/0'/0'/0/0   |
| Ethereum                | `ethereum`              | `0x...`        | m/44'/60'/0'/0/0  |
| TRON                    | `tron`                  | `T...`         | m/44'/195'/0'/0/0 |
| Binance Smart Chain     | `bsc`                   | `0x...`        | m/44'/60'/0'/0/0  |
| Litecoin                | `litecoin`              | `L...`, `M...` | m/44'/2'/0'/0/0   |
| Dogecoin                | `dogecoin`              | `D...`         | m/44'/3'/0'/0/0   |
| Solana                  | `solana`                | Base58         | m/44'/501'/0'/0/0 |

## Installation

### Prerequisites

- Python 3.8 or higher
- (Optional) NVIDIA GPU with CUDA support for GPU acceleration

### Setup

```bash
# Clone the repository
git clone https://github.com/maryayi/wallet-recovery.git
cd wallet-recovery

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional)

For GPU acceleration, install the additional dependencies:

```bash
# Install CUDA toolkit first (if not already installed)
# See: https://developer.nvidia.com/cuda-downloads

# Install GPU dependencies
pip install numpy numba

# Verify GPU is detected
python -c "from numba import cuda; print('GPU:', cuda.get_current_device().name if cuda.is_available() else 'Not available')"
```

## Usage

### Basic Syntax

```bash
python recover.py -p "PHRASE" -a "ADDRESS" -b BLOCKCHAIN
```

### Arguments

| Argument       | Short | Required | Description                                     |
| -------------- | ----- | -------- | ----------------------------------------------- |
| `--phrase`     | `-p`  | Yes      | Partial mnemonic with `?` for missing words     |
| `--address`    | `-a`  | Yes      | Known wallet address to match                   |
| `--blockchain` | `-b`  | Yes      | Blockchain type (see supported list)            |
| `--device`     | `-d`  | No       | Device: `auto`, `cpu`, or `gpu` (default: auto) |
| `--workers`    | `-w`  | No       | Number of CPU cores (default: all)              |
| `--passphrase` |       | No       | Optional BIP-39 passphrase                      |
| `--output`     | `-o`  | No       | Output file (default: `recovered_phrase.txt`)   |

### Examples

#### Recover 1 Missing Word (Ethereum)

```bash
python recover.py \
  -p "? abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x9858EfFD232B4033E47d90003D41EC34EcaEda94" \
  -b ethereum
```

#### Recover 2 Missing Words (Bitcoin)

```bash
python recover.py \
  -p "abandon ? abandon abandon abandon ? abandon abandon abandon abandon abandon about" \
  -a "1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA" \
  -b bitcoin
```

#### Recover Missing Word at Specific Position (TRON)

```bash
python recover.py \
  -p "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon ? about" \
  -a "TUEZSdKsoDHQMeZwihtdoBiN46zxhGWYdH" \
  -b tron
```

#### Using Custom Worker Count

```bash
python recover.py \
  -p "? abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x9858EfFD232B4033E47d90003D41EC34EcaEda94" \
  -b ethereum \
  -w 4
```

#### With BIP-39 Passphrase

```bash
python recover.py \
  -p "? abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x..." \
  -b ethereum \
  --passphrase "my secret passphrase"
```

#### Force GPU Mode

```bash
python recover.py \
  -p "? ? abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x9858EfFD232B4033E47d90003D41EC34EcaEda94" \
  -b ethereum \
  --device gpu
```

#### Force CPU Mode (disable GPU)

```bash
python recover.py \
  -p "? abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x9858EfFD232B4033E47d90003D41EC34EcaEda94" \
  -b ethereum \
  --device cpu
```

## Performance

Recovery time depends on the number of missing words:

| Missing Words | Combinations    | Estimated Time\* |
| ------------- | --------------- | ---------------- |
| 1             | 2,048           | Seconds          |
| 2             | 4,194,304       | Minutes          |
| 3             | 8,589,934,592   | Hours to Days    |
| 4+            | ~17.6 trillion+ | Impractical      |

\*Times vary based on hardware. Uses all CPU cores by default.

### Last Word Optimization

When only the **last word** is missing, the tool applies an entropy-based optimization that dramatically reduces the search space:

| Phrase Length | Standard Search | Optimized Search | Speedup |
| ------------- | --------------- | ---------------- | ------- |
| 12 words      | 2,048           | 128              | 16x     |
| 15 words      | 2,048           | 64               | 32x     |
| 18 words      | 2,048           | 32               | 64x     |
| 21 words      | 2,048           | 16               | 128x    |
| 24 words      | 2,048           | 8                | 256x    |

This optimization exploits BIP-39's checksum structure: the last word contains both entropy bits and checksum bits, so only specific words can produce a valid mnemonic given the other known words.

### GPU Acceleration

When a CUDA-capable GPU is available, the tool uses GPU parallelism for checksum validation:

- **Batch Processing**: Processes up to 1 million combinations per GPU batch
- **Parallel SHA-256**: Validates checksums in parallel across thousands of GPU cores
- **Hybrid Approach**: GPU handles checksum validation, CPU handles address derivation
- **Auto-detection**: Automatically detects GPU and falls back to CPU if unavailable

GPU mode is particularly effective for 2+ missing words where checksum validation becomes the bottleneck.

| Mode | Best For          | Notes                                       |
| ---- | ----------------- | ------------------------------------------- |
| CPU  | 1 missing word    | Low overhead, fast for small searches       |
| GPU  | 2-3 missing words | Massive parallelism for checksum validation |
| Auto | Any               | Automatically selects best available option |

## How It Works

1. **Parse Input**: Identifies missing word positions (marked with `?`)
2. **Entropy Optimization**: If only the last word is missing, pre-computes valid candidates using BIP-39 checksum constraints
3. **Generate Combinations**: Creates word combinations from the candidate list (optimized or full wordlist)
4. **Checksum Validation**: Quickly rejects invalid mnemonics (skipped when optimization is applied)
5. **Address Derivation**: Derives wallet address using blockchain-specific derivation path
6. **Match Verification**: Compares derived address with target address
7. **Output**: Saves recovered phrase to file when found

## Output

### Success

```
============================================================
SUCCESS! Recovered mnemonic phrase:
============================================================

  abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about

============================================================
Saved to: recovered_phrase.txt
```

### Failure

```
============================================================
FAILED: No matching phrase found
============================================================

Possible reasons:
  - Wrong blockchain selected
  - Wrong address
  - More words are missing than indicated
  - Non-standard derivation path used
```

## Security Considerations

- **Run Offline**: For maximum security, run this tool on an air-gapped computer
- **Clear History**: Delete recovered phrase files after use
- **Verify Source**: Only use this tool from trusted sources
- **Private Keys**: This tool never exposes private keys, only recovers the seed phrase

## Troubleshooting

### "No matching phrase found"

- Verify the blockchain type matches your wallet
- Ensure the address is correct (copy-paste, don't type)
- Check if the wallet uses a non-standard derivation path
- Confirm the number of missing words is correct

### Slow Performance

- Use all available CPU cores (default behavior)
- Enable GPU acceleration if you have an NVIDIA GPU
- Consider running on a machine with more cores
- Limit missing words to 3 or fewer

### Memory Issues

- Reduce worker count with `-w` flag
- Close other applications

### GPU Not Detected

- Ensure NVIDIA drivers are installed
- Install CUDA toolkit from NVIDIA
- Install `numpy` and `numba` packages
- Verify with: `python -c "from numba import cuda; print(cuda.is_available())"`

### GPU Out of Memory

- Reduce batch size (edit `batch_size` in code)
- Close other GPU applications
- Use `--device cpu` to fall back to CPU mode

## Dependencies

### Required

- [mnemonic](https://pypi.org/project/mnemonic/) - BIP-39 wordlist and validation
- [bip_utils](https://pypi.org/project/bip-utils/) - Multi-chain address derivation
- [tqdm](https://pypi.org/project/tqdm/) - Progress bar

### Optional (GPU Acceleration)

- [numpy](https://pypi.org/project/numpy/) - Numerical arrays for GPU data transfer
- [numba](https://pypi.org/project/numba/) - CUDA JIT compilation for GPU kernels
- NVIDIA CUDA Toolkit - GPU drivers and runtime

## License

MIT License - See [LICENSE](LICENSE) for details.

## Disclaimer

This tool is intended for recovering your own wallet seed phrases. Only use it on wallets you own. The authors are not responsible for any misuse or loss of funds.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
