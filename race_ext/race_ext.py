# race_ext.py
import os, platform
from torch.utils.cpp_extension import load

os.environ["OMP_NUM_THREADS"] = "16"

extra_cflags = ["-O3"]
extra_ldflags = []

if platform.system() == "Darwin":
    omp_prefix = os.environ.get("LIBOMP_PREFIX")
    if not omp_prefix:
        for p in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
            if os.path.isdir(p):
                omp_prefix = p; break
    if not omp_prefix:
        raise SystemExit("libomp not found. brew install libomp or set LIBOMP_PREFIX")
    extra_cflags += ["-Xpreprocessor", "-fopenmp", f"-I{omp_prefix}/include"]
    extra_ldflags += [f"-L{omp_prefix}/lib", "-lomp", f"-Wl,-rpath,{omp_prefix}/lib"]
else:
    extra_cflags += ["-fopenmp"]
    extra_ldflags += ["-fopenmp"]

race_pref = load(
    name="race_pref",
    sources=[os.path.join(os.path.dirname(__file__), "race_pref.cpp")], # Path to the .cpp file
    extra_cflags=extra_cflags,
    extra_ldflags=extra_ldflags,
    verbose=True,  # keep it quiet once it’s cached
)

# linear_pref = load(
#     name="linear_pref",
#     sources=["/Users/sahiljoshi/Documents/Research/linear_pref.cpp"], # Path to the .cpp file
#     extra_cflags=extra_cflags,
#     extra_ldflags=extra_ldflags,
#     verbose=True,  # keep it quiet once it’s cached
# )
