# GPU-based parallel implementation of the permutation testing in NIST SP 800-90B

## Requirements
* Platform : Winodws 10
* Visual Studio 2017
* Cuda compilation tools, release 10.1, V10.1.105
* GPU have at least 2.5GB or more of the global memory.

## Overview
* `test_data/` has binary files for testing.
* `cpp/` holds the codebase.

## How to run
Load the projects in Visual Studio, and then run the program (press Ctrl+F5 (Start without debugging) or use the green Start button on the Visual Studio toolbar). 

Alternatively, run the program on the windows command prompt:
<pre><code>cd /path/to/cuda_iid/x64/Release
cuda_iid.exe
</code></pre>

* __The IID noise source__
![iid](https://user-images.githubusercontent.com/65601912/99964203-493e4100-2dd6-11eb-9f32-ff79b0f00a21.JPG)

* __The non-IID noise source__
![non_iid](https://user-images.githubusercontent.com/65601912/99964296-6bd05a00-2dd6-11eb-976f-a3f594275dc7.JPG)

