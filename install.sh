python setup_cython.py build_ext --inplace
mv src/GCMProject/state_evolution/auxiliary/cython_functions.cpython-39-darwin.so /Users/clarte/opt/usr/local/lib
python -m build -n
pip install dist/gcmclarte-0.0.1-py3-none-any.whl --force-reinstall