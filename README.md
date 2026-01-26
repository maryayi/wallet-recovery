# Wallet Recovery

A Python tool to recover missing words from BIP-39 mnemonic seed phrases by brute-forcing possible combinations and validating against a known wallet address.

## Features

- **Multi-Blockchain Support**: Bitcoin, Ethereum, TRON, BSC, Litecoin, Dogecoin, Solana
- **Multiple Bitcoin Address Formats**: Legacy, SegWit, Native SegWit
- **GPU Acceleration**: CUDA (NVIDIA) and OpenCL (Intel/AMD) support for massively parallel checksum validation
- **CPU Fallback**: Automatically falls back to multi-core CPU if no GPU available
- **Fast Validation**: Checksum validation before address derivation for quick rejection
- **Parallel Processing**: Multi-core CPU support for faster recovery
- **Progress Tracking**: Real-time progress bar with completion estimates
- **Resumable Sessions**: Interrupt and resume long-running recoveries; state is auto-saved periodically
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
- (Optional) GPU for acceleration:
  - NVIDIA GPU with CUDA support, or
  - Intel/AMD GPU with OpenCL support

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

#### NVIDIA GPU (CUDA)

```bash
# Install CUDA toolkit first (if not already installed)
# See: https://developer.nvidia.com/cuda-downloads

# Install CUDA dependencies
pip install numpy numba

# Verify CUDA is detected
python -c "from numba import cuda; print('CUDA GPU:', cuda.get_current_device().name if cuda.is_available() else 'Not available')"
```

#### Intel/AMD GPU (OpenCL)

```bash
# Install OpenCL runtime
# Ubuntu/Debian:
sudo apt install intel-opencl-icd    # For Intel GPUs
# or
sudo apt install mesa-opencl-icd     # For AMD GPUs

# Install OpenCL dependencies
pip install numpy pyopencl

# Verify OpenCL is detected
python -c "import pyopencl as cl; print('OpenCL devices:', [d.name.strip() for p in cl.get_platforms() for d in p.get_devices()])"
```

## Usage

### Basic Syntax

```bash
python recover.py -p "PHRASE" -a "ADDRESS" -b BLOCKCHAIN
```

### Arguments

| Argument        | Short | Required | Description                                                       |
| --------------- | ----- | -------- | ----------------------------------------------------------------- |
| `--phrase`      | `-p`  | Yes\*    | Partial mnemonic with `?` for missing words                       |
| `--address`     | `-a`  | Yes\*    | Known wallet address to match                                     |
| `--blockchain`  | `-b`  | Yes\*    | Blockchain type (see supported list)                              |
| `--device`      | `-d`  | No       | Device: `auto`, `cpu`, `gpu`, `cuda`, or `opencl` (default: auto) |
| `--workers`     | `-w`  | No       | Number of CPU cores (default: all)                                |
| `--passphrase`  |       | No       | Optional BIP-39 passphrase                                        |
| `--output`      | `-o`  | No       | Output file (default: `recovered_phrase.txt`)                     |
| `--resume`      |       | No       | Resume from a previously interrupted session                      |
| `--state-file`  |       | No       | Custom state file path (default: `.wallet_recovery_state.json`)   |
| `--clear-state` |       | No       | Clear saved state and start fresh                                 |
| `--show-state`  |       | No       | Display saved state information and exit                          |

\* Not required when using `--resume` (values are loaded from saved state)

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

#### Force GPU Mode (auto-select best available)

```bash
python recover.py \
  -p "? ? abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x9858EfFD232B4033E47d90003D41EC34EcaEda94" \
  -b ethereum \
  --device gpu
```

#### Force CUDA (NVIDIA GPU)

```bash
python recover.py \
  -p "? ? abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x9858EfFD232B4033E47d90003D41EC34EcaEda94" \
  -b ethereum \
  --device cuda
```

#### Force OpenCL (Intel/AMD GPU)

```bash
python recover.py \
  -p "? ? abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x9858EfFD232B4033E47d90003D41EC34EcaEda94" \
  -b ethereum \
  --device opencl
```

#### Force CPU Mode (disable GPU)

```bash
python recover.py \
  -p "? abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x9858EfFD232B4033E47d90003D41EC34EcaEda94" \
  -b ethereum \
  --device cpu
```

#### Resume an Interrupted Session

```bash
# If you interrupt a running session with Ctrl+C, resume with:
python recover.py --resume

# Resume from a custom state file:
python recover.py --resume --state-file my_session.json
```

#### View Saved State Information

```bash
python recover.py --show-state
```

#### Clear Saved State and Start Fresh

```bash
python recover.py \
  -p "? abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" \
  -a "0x9858EfFD232B4033E47d90003D41EC34EcaEda94" \
  -b ethereum \
  --clear-state
```

## State Management (Resumable Sessions)

The tool automatically saves progress periodically, allowing you to interrupt and resume long-running recovery sessions.

### How It Works

- **Auto-save**: Progress is saved every 10,000 combinations to `.wallet_recovery_state.json`
- **Graceful Interrupts**: Pressing `Ctrl+C` saves state before exiting
- **Resume**: Use `--resume` to continue from where you left off
- **Auto-cleanup**: State is automatically cleared on success or when all combinations are exhausted

### State File Contents

The state file stores:

- Partial phrase, target address, blockchain, and passphrase
- Device type (cpu/cuda/opencl)
- Number of processed combinations and total combinations
- Session ID and elapsed time

### Example Workflow

```bash
# Start a recovery session
python recover.py -p "? ? abandon abandon ..." -a "0x..." -b ethereum

# Press Ctrl+C to interrupt
# Output: "State saved. Processed 1,234,567 of 4,194,304 combinations."

# Later, resume the session
python recover.py --resume
# Output: "Resuming previous session (ID: a1b2c3d4e5f6)"
# Output: "Progress: 1,234,567 / 4,194,304 (29.4%)"

# Check state without running
python recover.py --show-state
```

### Multiple Sessions

Use `--state-file` to manage multiple recovery sessions:

```bash
# Start first session
python recover.py -p "..." -a "0x111..." -b ethereum --state-file session1.json

# Start second session (different wallet)
python recover.py -p "..." -a "0x222..." -b bitcoin --state-file session2.json

# Resume specific session
python recover.py --resume --state-file session1.json
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

The tool supports two GPU backends for parallel checksum validation:

#### Supported GPU Backends

| Backend | GPU Support        | Library  | Notes                                      |
| ------- | ------------------ | -------- | ------------------------------------------ |
| CUDA    | NVIDIA             | numba    | Best performance on NVIDIA GPUs            |
| OpenCL  | Intel, AMD, NVIDIA | pyopencl | Cross-platform, works with integrated GPUs |

#### How It Works

- **Batch Processing**: Processes up to 1 million combinations per GPU batch
- **Parallel SHA-256**: Validates checksums in parallel across thousands of GPU cores
- **Hybrid Approach**: GPU handles checksum validation, CPU handles address derivation
- **Auto-detection**: Automatically detects available GPUs and selects the best backend

GPU mode is particularly effective for 2+ missing words where checksum validation becomes the bottleneck.

#### Device Selection

| Mode     | Best For       | Notes                                             |
| -------- | -------------- | ------------------------------------------------- |
| `auto`   | Any            | Auto-selects best available (CUDA > OpenCL > CPU) |
| `gpu`    | Any GPU        | Uses best available GPU backend                   |
| `cuda`   | NVIDIA GPUs    | Forces CUDA backend                               |
| `opencl` | Intel/AMD GPUs | Forces OpenCL backend                             |
| `cpu`    | 1 missing word | Low overhead, fast for small searches             |

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

### CUDA GPU Not Detected

- Ensure NVIDIA drivers are installed
- Install CUDA toolkit from NVIDIA
- Install `numpy` and `numba` packages
- Verify with: `python -c "from numba import cuda; print(cuda.is_available())"`

### OpenCL GPU Not Detected

- Install OpenCL runtime for your GPU:
  - **Intel**: `sudo apt install intel-opencl-icd`
  - **AMD**: `sudo apt install mesa-opencl-icd` or install ROCm
  - **NVIDIA**: OpenCL is included with NVIDIA drivers
- Install `numpy` and `pyopencl` packages
- Verify with: `python -c "import pyopencl as cl; print([d.name for p in cl.get_platforms() for d in p.get_devices()])"`

### GPU Out of Memory

- Reduce batch size (edit `batch_size` in code)
- Close other GPU applications
- Use `--device cpu` to fall back to CPU mode

### State/Resume Issues

- **"No saved state found"**: No previous session exists; start a new one with `-p`, `-a`, `-b`
- **State file corrupted**: Delete `.wallet_recovery_state.json` and start fresh
- **Wrong session resumed**: Use `--show-state` to verify, or use `--state-file` to specify the correct file
- **Want to start over**: Use `--clear-state` flag to remove saved state

## Dependencies

### Required

- [mnemonic](https://pypi.org/project/mnemonic/) - BIP-39 wordlist and validation
- [bip_utils](https://pypi.org/project/bip-utils/) - Multi-chain address derivation
- [tqdm](https://pypi.org/project/tqdm/) - Progress bar
- [numpy](https://pypi.org/project/numpy/) - Numerical arrays for GPU data transfer

### Optional (CUDA - NVIDIA GPUs)

- [numba](https://pypi.org/project/numba/) - CUDA JIT compilation for GPU kernels
- NVIDIA CUDA Toolkit - GPU drivers and runtime

### Optional (OpenCL - Intel/AMD GPUs)

- [pyopencl](https://pypi.org/project/pyopencl/) - OpenCL bindings for Python
- OpenCL runtime:
  - Intel: `intel-opencl-icd` package
  - AMD: `mesa-opencl-icd` or ROCm
  - NVIDIA: Included with drivers

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
