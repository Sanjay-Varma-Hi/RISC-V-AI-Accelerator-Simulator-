# RISC-V AI Accelerator Simulator

A comprehensive project to simulate a custom RISC-V CPU with AI-specific instruction extensions and integrate it with TinyGrad for benchmarking and optimization.

## 🚀 Project Overview

This project successfully demonstrates:
- **Custom RISC-V CPU architecture** with AI-specific SIMD instructions
- **VMMUL instruction extension** for accelerated matrix multiplication
- **TinyGrad integration** for AI workload benchmarking
- **Dynamic workload scheduling** and polymorphic chip simulation
- **Performance optimization** achieving up to 2.14x speedup

## 📁 Project Structure

```
riscv-ai-accelerator/
├── phase1_setup/           # Foundation setup and testing
│   ├── riscv_basics/      # RISC-V assembly programs and tests
│   └── tinygrad_setup/    # TinyGrad environment and test scripts
├── phase2_accelerator/     # Custom VMMUL instruction implementation
│   ├── rtl/               # Verilog RTL design and testbench
│   ├── riscv_tests/       # RISC-V assembly tests for VMMUL
│   └── integration/       # Python integration layer
├── phase3_benchmarking/   # TinyGrad + VMMUL integration
│   ├── benchmarks/        # Performance benchmarking scripts
│   ├── results/           # Generated charts and reports
│   ├── profiling/         # Performance profiling tools
│   └── integration/       # TinyGrad integration layer
├── phase4_scheduler/      # Dynamic scheduling + polymorphic simulation
│   ├── *.py              # Core scheduling and simulation scripts
│   ├── *.png             # Performance visualization charts
│   ├── *.csv             # Benchmark and simulation data
│   └── *.md              # Documentation and demo scripts
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎯 Development Phases

### ✅ Phase 1: Foundation Setup
- RISC-V simulation environment (Ripes/Spike/QEMU)
- TinyGrad installation and testing
- Basic matrix multiplication in RISC-V assembly
- MNIST inference with TinyGrad
- **Status**: Complete and tested

### ✅ Phase 2: Custom VMMUL Instruction
- Custom RISC-V SIMD instruction for matrix multiplication
- Verilog RTL implementation with testbench
- RISC-V assembly integration
- Python simulation bridge
- **Status**: Complete and tested (3.5x-7x speedup achieved)

### ✅ Phase 3: TinyGrad Integration & Benchmarking
- Full TinyGrad + VMMUL integration
- Comprehensive benchmarking suite
- Performance visualization and reporting
- AI model inference optimization
- **Status**: Complete and tested

### ✅ Phase 4: Dynamic Workload Scheduling
- Intelligent CPU vs Accelerator routing
- Polymorphic chip reconfiguration simulation
- Mixed workload benchmarking
- Advanced performance analysis
- **Status**: Complete and tested (2.14x speedup achieved)

## 🏆 Key Achievements

- **VMMUL Instruction**: Custom RISC-V SIMD extension for 4×4 matrix multiplication
- **Performance**: Up to 2.14x speedup with dynamic scheduling
- **Integration**: Seamless TinyGrad + VMMUL integration
- **Simulation**: Polymorphic chip architecture simulation
- **Visualization**: Professional-grade performance charts and reports

## 🚀 Getting Started

1. **RISC-V Setup**: Follow instructions in `phase1_setup/riscv_basics/`
2. **TinyGrad Setup**: Follow instructions in `phase1_setup/tinygrad_setup/`
3. **VMMUL Testing**: Run tests in `phase2_accelerator/integration/`
4. **Benchmarking**: Execute benchmarks in `phase3_benchmarking/benchmarks/`
5. **Dynamic Scheduling**: Test scheduler in `phase4_scheduler/`

## 📊 Quick Performance Test

```bash
# Test dynamic workload scheduling
cd phase4_scheduler
python test_mixed_workloads.py

# View performance charts
python generate_phase4_graphs.py
open *.png
```

## 🔧 Requirements

- **OS**: macOS (Apple Silicon M1/M2 recommended)
- **Python**: 3.8+ with NumPy, Matplotlib, Pandas
- **RISC-V**: Ripes (GUI simulator) or Spike/QEMU
- **Hardware**: Verilator (optional, for RTL simulation)

## 📈 Performance Results

| Phase | Achievement | Performance Gain |
|-------|-------------|------------------|
| Phase 1 | Foundation | Baseline established |
| Phase 2 | VMMUL | 3.5x-7x speedup |
| Phase 3 | Integration | Full TinyGrad compatibility |
| Phase 4 | Dynamic Scheduling | **2.14x speedup** |

## 🎯 Business Impact

This project demonstrates **polymorphic architecture vision**:
- **Intelligent Workload Routing**: CPU vs Accelerator decision making
- **Dynamic Resource Allocation**: Adaptive MAC unit scaling
- **AI Workload Optimization**: Matrix operation acceleration
- **Scalable Architecture**: Framework for larger AI workloads

## 📚 Documentation

- **Phase 1**: `PHASE1_COMPLETION_SUMMARY.md`
- **Phase 2**: `PHASE2_COMPLETION_SUMMARY.md`
- **Phase 3**: `PHASE3_COMPLETION_SUMMARY.md`
- **Phase 4**: `phase4_scheduler/README.md`

## 🤝 Contributing

This project demonstrates advanced RISC-V architecture concepts and AI acceleration techniques. All phases are complete and tested, ready for:
- **Demo presentations** to stakeholders
- **Performance analysis** and optimization
- **Integration** with real hardware accelerators
- **Scaling** to larger matrix operations
