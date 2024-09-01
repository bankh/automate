Using pip 24.0 from /opt/conda/lib/python3.7/site-packages/pip (python 3.7)
Non-user install because site-packages writeable
Created temporary directory: /tmp/pip-build-tracker-844x3qky
Initialized build tracking at /tmp/pip-build-tracker-844x3qky
Created build tracker: /tmp/pip-build-tracker-844x3qky
Entered build tracker: /tmp/pip-build-tracker-844x3qky
Created temporary directory: /tmp/pip-install-blo5utc_
Created temporary directory: /tmp/pip-ephem-wheel-cache-4i6buohp
Processing /media/ubuntu/phd_thesis/software/deGravity_jones/automate
  Added file:///media/ubuntu/phd_thesis/software/deGravity_jones/automate to build tracker '/tmp/pip-build-tracker-844x3qky'
  Running setup.py (path:/media/ubuntu/phd_thesis/software/deGravity_jones/automate/setup.py) egg_info for package from file:///media/ubuntu/phd_thesis/software/deGravity_jones/automate
  Created temporary directory: /tmp/pip-pip-egg-info-jp75c7u4
  Preparing metadata (setup.py): started
  Running command python setup.py egg_info
  running egg_info
  creating /tmp/pip-pip-egg-info-jp75c7u4/automate.egg-info
  writing /tmp/pip-pip-egg-info-jp75c7u4/automate.egg-info/PKG-INFO
  writing dependency_links to /tmp/pip-pip-egg-info-jp75c7u4/automate.egg-info/dependency_links.txt
  writing top-level names to /tmp/pip-pip-egg-info-jp75c7u4/automate.egg-info/top_level.txt
  writing manifest file '/tmp/pip-pip-egg-info-jp75c7u4/automate.egg-info/SOURCES.txt'
  reading manifest file '/tmp/pip-pip-egg-info-jp75c7u4/automate.egg-info/SOURCES.txt'
  writing manifest file '/tmp/pip-pip-egg-info-jp75c7u4/automate.egg-info/SOURCES.txt'
  Preparing metadata (setup.py): finished with status 'done'
  Source in /media/ubuntu/phd_thesis/software/deGravity_jones/automate has version 1.0.1, which satisfies requirement automate==1.0.1 from file:///media/ubuntu/phd_thesis/software/deGravity_jones/automate
  Removed automate==1.0.1 from file:///media/ubuntu/phd_thesis/software/deGravity_jones/automate from build tracker '/tmp/pip-build-tracker-844x3qky'
Created temporary directory: /tmp/pip-unpack-fx2dass6
Building wheels for collected packages: automate
  Created temporary directory: /tmp/pip-wheel-wdz6zh9w
  Building wheel for automate (setup.py): started
  Destination directory: /tmp/pip-wheel-wdz6zh9w
  Running command python setup.py bdist_wheel
  running bdist_wheel
  running build
  running build_py
  creating build
  creating build/lib.linux-x86_64-cpython-37
  creating build/lib.linux-x86_64-cpython-37/automate
  copying automate/__init__.py -> build/lib.linux-x86_64-cpython-37/automate
  copying automate/sbgcn.py -> build/lib.linux-x86_64-cpython-37/automate
  copying automate/util.py -> build/lib.linux-x86_64-cpython-37/automate
  copying automate/arg_parsing.py -> build/lib.linux-x86_64-cpython-37/automate
  copying automate/uvnet_encoders.py -> build/lib.linux-x86_64-cpython-37/automate
  copying automate/eclasses.py -> build/lib.linux-x86_64-cpython-37/automate
  copying automate/pointnet_encoder.py -> build/lib.linux-x86_64-cpython-37/automate
  copying automate/brep.py -> build/lib.linux-x86_64-cpython-37/automate
  copying automate/plot_confusion_matrix.py -> build/lib.linux-x86_64-cpython-37/automate
  copying automate/conversions.py -> build/lib.linux-x86_64-cpython-37/automate
  running build_ext
  cmake /media/ubuntu/phd_thesis/software/deGravity_jones/automate -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=/media/ubuntu/phd_thesis/software/deGravity_jones/automate/build/lib.linux-x86_64-cpython-37 -DCMAKE_BUILD_TYPE=Release
  -- The C compiler identification is GNU 9.4.0
  -- The CXX compiler identification is GNU 9.4.0
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Check for working C compiler: /usr/bin/cc - skipped
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Check for working CXX compiler: /usr/bin/c++ - skipped
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- Found PythonInterp: /opt/conda/bin/python (found version "3.7.7")
  -- Found PythonLibs: /opt/conda/lib/libpython3.7m.so
  -- Performing Test HAS_FLTO
  -- Performing Test HAS_FLTO - Success
  -- Found pybind11: /opt/conda/include (found version "2.6.1" )
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /media/ubuntu/phd_thesis/software/deGravity_jones/automate/build/temp.linux-x86_64-cpython-37
  cmake --build . --config Release
  [  7%] Building CXX object _deps/brloader-build/CMakeFiles/breploader.dir/src/body.cpp.o
  [ 15%] Building CXX object _deps/brloader-build/CMakeFiles/breploader.dir/src/occ/occtbody.cpp.o
  [ 23%] Building CXX object _deps/brloader-build/CMakeFiles/breploader.dir/src/occ/occtedge.cpp.o
  [ 30%] Building CXX object _deps/brloader-build/CMakeFiles/breploader.dir/src/occ/occtface.cpp.o
  [ 38%] Building CXX object _deps/brloader-build/CMakeFiles/breploader.dir/src/occ/occtloop.cpp.o
  [ 46%] Building CXX object _deps/brloader-build/CMakeFiles/breploader.dir/src/occ/occtvertex.cpp.o
  [ 53%] Linking CXX static library libbreploader.a
  [ 53%] Built target breploader
  [ 61%] Building CXX object CMakeFiles/automate_cpp.dir/cpp/automate.cpp.o
  [ 69%] Building CXX object CMakeFiles/automate_cpp.dir/cpp/disjointset.cpp.o
  [ 76%] Building CXX object CMakeFiles/automate_cpp.dir/cpp/lsh.cpp.o
  [ 84%] Building CXX object CMakeFiles/automate_cpp.dir/cpp/eclass.cpp.o
  [ 92%] Building CXX object CMakeFiles/automate_cpp.dir/cpp/part.cpp.o
  [100%] Linking CXX shared module ../lib.linux-x86_64-cpython-37/automate_cpp.cpython-37m-x86_64-linux-gnu.so
  [100%] Built target automate_cpp
  /opt/conda/lib/python3.7/site-packages/setuptools/_distutils/cmd.py:66: SetuptoolsDeprecationWarning: setup.py install is deprecated.
  !!

          ********************************************************************************
          Please avoid running ``setup.py`` directly.
          Instead, use pypa/build, pypa/installer or other
          standards-based tools.

          See https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html for details.
          ********************************************************************************

  !!
    self.initialize_options()
  installing to build/bdist.linux-x86_64/wheel
  running install
  running install_lib
  creating build/bdist.linux-x86_64
  creating build/bdist.linux-x86_64/wheel
  creating build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/__init__.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/sbgcn.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/util.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/arg_parsing.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/uvnet_encoders.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/eclasses.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/pointnet_encoder.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/brep.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/plot_confusion_matrix.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate/conversions.py -> build/bdist.linux-x86_64/wheel/automate
  copying build/lib.linux-x86_64-cpython-37/automate_cpp.cpython-37m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/wheel
  running install_egg_info
  running egg_info
  creating automate.egg-info
  writing automate.egg-info/PKG-INFO
  writing dependency_links to automate.egg-info/dependency_links.txt
  writing top-level names to automate.egg-info/top_level.txt
  writing manifest file 'automate.egg-info/SOURCES.txt'
  reading manifest file 'automate.egg-info/SOURCES.txt'
  writing manifest file 'automate.egg-info/SOURCES.txt'
  Copying automate.egg-info to build/bdist.linux-x86_64/wheel/automate-1.0.1-py3.7.egg-info
  running install_scripts
  creating build/bdist.linux-x86_64/wheel/automate-1.0.1.dist-info/WHEEL
  creating '/tmp/pip-wheel-wdz6zh9w/automate-1.0.1-cp37-cp37m-linux_x86_64.whl' and adding 'build/bdist.linux-x86_64/wheel' to it
  adding 'automate_cpp.cpython-37m-x86_64-linux-gnu.so'
  adding 'automate/__init__.py'
  adding 'automate/arg_parsing.py'
  adding 'automate/brep.py'
  adding 'automate/conversions.py'
  adding 'automate/eclasses.py'
  adding 'automate/plot_confusion_matrix.py'
  adding 'automate/pointnet_encoder.py'
  adding 'automate/sbgcn.py'
  adding 'automate/util.py'
  adding 'automate/uvnet_encoders.py'
  adding 'automate-1.0.1.dist-info/METADATA'
  adding 'automate-1.0.1.dist-info/WHEEL'
  adding 'automate-1.0.1.dist-info/top_level.txt'
  adding 'automate-1.0.1.dist-info/RECORD'
  removing build/bdist.linux-x86_64/wheel
  Building wheel for automate (setup.py): finished with status 'done'
  Created wheel for automate: filename=automate-1.0.1-cp37-cp37m-linux_x86_64.whl size=297217 sha256=166ded638343df50685d3b9796c9735531f21e8868497f0e2e2f2b141d355be8
  Stored in directory: /tmp/pip-ephem-wheel-cache-4i6buohp/wheels/ee/aa/4c/e50ce302145cb327fe45edb7c9c93f628a669bdd68b170e61c
Successfully built automate
DEPRECATION: pytorch-lightning 1.7.0 has a non-standard dependency specifier torch>=1.9.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
Installing collected packages: automate
  Attempting uninstall: automate
    Found existing installation: automate 1.0.1
    Uninstalling automate-1.0.1:
      Created temporary directory: /opt/conda/lib/python3.7/site-packages/~utomate-1.0.1.dist-info
      Removing file or directory /opt/conda/lib/python3.7/site-packages/automate-1.0.1.dist-info/
      Created temporary directory: /opt/conda/lib/python3.7/site-packages/~utomate
      Removing file or directory /opt/conda/lib/python3.7/site-packages/automate/
      Created temporary directory: /tmp/pip-uninstall-r705v18i
      Removing file or directory /opt/conda/lib/python3.7/site-packages/automate_cpp.cpython-38-x86_64-linux-gnu.so
      Successfully uninstalled automate-1.0.1

Successfully installed automate-1.0.1
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
Remote version of pip: 24.0
Local version of pip:  20.1.1
Was pip installed by pip? True

[notice] A new release of pip is available: 20.1.1 -> 24.0
[notice] To update, run: pip install --upgrade pip
Removed build tracker: '/tmp/pip-build-tracker-844x3qky'
