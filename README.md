# GPU-based parallel implementation of the permutation testing in NIST SP 800-90B.

## Requirements
* Platform : Winodws 10
* Visual Studio 2017
* Cuda compilation tools, release 10.1, V10.1.105
* GPU have at least 2GB or more of the global memory.

## Overview
* `<test_data/>` has binary files for testing.
* `<cpp/>` holds the codebase.

## How to run
Load the projects in Visual Studio, and then run the program (press Ctrl+F5 (Start without debugging) or use the green Start button on the Visual Studio toolbar). 

Alternatively, run the program on the windows command prompt:
<pre><code>cd /path/to/GPU_based_parallel_permutation_testing/x64/Release
CUDA_IID_PERM.exe
</code></pre>

* The IID noise source

* The non-IID noise source

