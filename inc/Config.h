#define NOTE(string)

NOTE("SIMTight SoC configuration")

NOTE("DRAM configuration")
NOTE("==================")

NOTE("Size of data field in DRAM request/response")
#define DRAMBeatBits 512
#define DRAMBeatBytes 64
#define DRAMBeatHalfs 32
#define DRAMBeatWords 16
#define DRAMBeatLogBytes 6

NOTE("Max burst size = 2^(DRAMBurstWidth-1)")
#define DRAMBurstWidth 4

NOTE("Size of DRAM in beats")
#define DRAMAddrWidth 26

NOTE("Maximum number of inflight requests supported by DRAM wrapper")
#define DRAMLogMaxInFlight 5

NOTE("SIMT configuration")
NOTE("==================")

NOTE("Number of lanes per SIMT core")
#define SIMTLanes 32
#define SIMTLogLanes 5

NOTE("Number of warps per SIMT core")
#define SIMTWarps 64
#define SIMTLogWarps 6

NOTE("Number of bits used to track divergence nesting level")
#define SIMTLogMaxNestLevel 5

NOTE("Stack size (in bytes) for each SIMT thread")
#define SIMTLogBytesPerStack 19

NOTE("Number of SRAM banks")
#define SIMTLogSRAMBanks 4
#define SIMTSRAMBanks 16

NOTE("Size of each SRAM bank (in words)")
#define SIMTLogWordsPerSRAMBank 10

NOTE("Enable SIMT stat counters")
#define SIMTEnableStatCounters 1

NOTE("Use full-throughput divider (rather than sequential divider)?")
#define SIMTUseFullDivider 0

NOTE("Latency of full-throughput divider (more latency = higher Fmax)")
#define SIMTFullDividerLatency 12

NOTE("Use scalarising register file?")
#define SIMTEnableRegFileScalarisation 0

NOTE("Use affine scalarisation, or just plain uniform scalarisation?")
#define SIMTEnableAffineScalarisation 0

NOTE("For affine scalarisation, how many bits to use for the stride?")
#define SIMTAffineScalarisationBits 4

NOTE("Size of scalarising register file (number of vectors)")
#define SIMTRegFileSize 2048

NOTE("Use dedicated scalar unit, allowing parallel scalar/vector execution?")
#define SIMTEnableScalarUnit 0

NOTE("[EXPERIMENTAL] Enable scalarised vector store buffer?")
#define SIMTEnableSVStoreBuffer 0

NOTE("Size of scalarised vector store buffer")
#define SIMTSVStoreBufferLogSize 11

NOTE("Use LRU register spilling policy based on approx mean reg usage")
#define SIMTUseLRUSpill 0

NOTE("For LRU spill, how many bits to use for mean?")
#define SIMTRegCountBits 8

NOTE("CPU configuration")
NOTE("=================")

NOTE("Size of tightly coupled instruction memory")
#define CPUInstrMemLogWords 13

NOTE("Number of cache lines (line size == DRAM beat size)")
#define SBDCacheLogLines 9

NOTE("Use register forwarding for increased IPC (but possibly lower Fmax)?")
#define CPUEnableRegForwarding 0

NOTE("Size of CPU's stack")
#define CPUStackSize 1073741824

NOTE("Tagged memory")
NOTE("=============")

NOTE("Is tagged memory enabled? (Needed for CHERI)")
#define EnableTaggedMem 0

NOTE("Tag cache: line size")
#define TagCacheLogBeatsPerLine 1

NOTE("Tag cache: number of set-associative ways")
#define TagCacheLogNumWays 2

NOTE("Tag cache: number of sets")
#define TagCacheLogSets 7

NOTE("Tag cache: max number of inflight memory requests")
#define TagCacheLogMaxInflight 5

NOTE("Tag cache: max number of pending requests per way")
#define TagCachePendingReqsPerWay 16

NOTE("Tag cache: optimise caching of large regions of zero tags")
#define TagCacheHierarchical 1

NOTE("CHERI support")
NOTE("=============")

NOTE("Is CHERI enabled? (If so, see UseClang and EnableTaggedMem settings)")
#define EnableCHERI 0

NOTE("Use scalarising register file for capability meta-data?")
#define SIMTEnableCapRegFileScalarisation 0

NOTE("Size of scalarising capability register file (number of vectors)")
#define SIMTCapRegFileSize 2048

NOTE("Use shared immutable PCC meta-data for all threads in kernel?")
#define SIMTUseSharedPCC 1

NOTE("Number of bounds-setting units")
#define SIMTNumSetBoundsUnits SIMTLanes

NOTE("Compiler")
NOTE("========")

NOTE("Use clang rather than gcc? (Currently required if CHERI enabled)")
#define UseClang 0

NOTE("Memory map")
NOTE("==========")

NOTE("Memory base (after tag bit region)")
#define MemBase 134217728

NOTE("Space reserved in instruction memory for boot loader")
#define MaxBootImageBytes 1024
