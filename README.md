# GPU-based parallel implementation of the permutation testing in NIST SP 800-90B

## Requirements
* Platform : Winodws 10
* Visual Studio 2017
* Cuda compilation tools, release 10.1, V10.1.105
* GPU have at least 2GB or more of the global memory.

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
![Result_IID_noise_source](https://user-images.githubusercontent.com/65601912/82523667-d8d50f80-9b67-11ea-912e-a045a250eb9c.JPG)

* __The non-IID noise source__
![Result_non_IID_noise_source](https://user-images.githubusercontent.com/65601912/82523680-e4c0d180-9b67-11ea-8948-e62d5f070e6a.JPG)

