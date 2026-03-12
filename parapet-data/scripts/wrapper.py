import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq.py")]
runpy.run_path(sys.argv[0], run_name="__main__")
