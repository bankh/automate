<div align="center" style="font-size: 22pt;"> 
<h1 style="text-align: center;">AutoMate</h1>
</div>

Accompanying dataset and code to the publication: [AutoMate: A Dataset and Learning Approach for Automatic Mating of CAD Assemblies](https://dl.acm.org/doi/10.1145/3478513.3480562) or arXiv prePrint: [2302.05533](https://arxiv.org/abs/2105.12238).

__Notes/ Changes from the Original Repository:__
- Great work on BRep structure in terms of the data and proposed model.
- The author already changed the dependency of Parasolid as optional and one can use an open-source BRep framework (e.g., [OpenCascade](https://github.com/Open-Cascade-SAS/OCCT)).
- The instructions on data (added [here](#data-preparation)) is not directly on the repository and training (added [here](#training-of-the-model)) is missing. 
- Some of the dependencies might be problematic in some systems (e.g., AMD-based GPUs) and it might be great to add some notes on the original system's features beyond the provided `.yml` files. Details of the hardware system and the installation of the dependencies are added [here](#requirements)---please expand `Added Installation on AMD-based System` for more details.

[ ] Multi-GPU training

## Dataset
The AutoMate dataset can be downloaded from [Zenodo](https://zenodo.org/record/7776208#.ZDcYinbMIQ8).

### Details of the Dataset
__from [README.md](https://zenodo.org/records/7776208/files/README.md?download=1) of Zenodo__  
The AutoMate dataset contains 451,967 unique CAD parts, 255,211 CAD assemblies, and 1,292,016 unique mates scrapped from public OnShape documents for the presented paper.  
The dataset is provided in both the original Parasolid format, as well as STEP files. Note that automatic conversion to STEP is not perfect, so some parts may be missing or have bad geometry. Missing parts have been annotated in the accompanying metadata which is stored as Apache parquet files, an open format and can be read by a variety of packages including `pandas`. Below are the details of each component of the dataset:  
<details>
    <summary><strong>1-Assembly JSONS</strong></summary>
    Assembly information is stored as JSON files with the following schema:
    <pre>
    {
        'assemblyId': str, // name of assembly in assemblies.zip; read as 'assemblies/{assemblyId}.json'
        'has_all_parasolid': bool, // if all parts in the assembly are in parasolid.zip
        'has_all_step':bool, // if all parts in the assembly are in step.zip
        parts:[
            {
                'id': string, // name of part in corresponding zip file; read as either 'step/{id}.step' or 'parasolid/{id}.x_t'
                'has_parasolid': bool, // if the part is present is parasolid.zip
                'has_step': bool, // if the part is present is step.zip 
            }, ...
        ],
        'occurrences': [
            {
                'part': int, // index into parts list
                'id': string, // unique id within assembly
                'transform': list[float], // flattened (row major) 4x4 homogenous transform matrix of part within assembly
                'fixed': bool, // if occurrence is constrained to be unmoving within assembly
                'hidden': bool, // if occurrence was hidden (invisible) in assembly
                'has_parasolid': bool, // if referenced part is in parasolid.zip
                'has_step': bool // if referenced part is in step.zip
            }, ...
        ],
        'mates': [
            {
                'name': str, // name of mate -- may not be unique
                'id': str, // unique id of mate within assembly
                'mateType': str, // type of constraint mate represents, see paper for descriptions
                'occurrences': list[int], // indices into occurrence list of constrained occurrences
                'mcfs': list[list[float]], // 2, 4x4 homogeneous frame matrices for mate connecting frames (see paper for description)
                'has_parasolid': bool, // if both referenced parts are in parasolid.zip
                'has_step': bool // if both referenced parts are in step.zip
            }, ...
        ],
        'mateRelations': [
            {
                'name': str, // name of mate relation -- may not be unique
                'relationType': str, // type of mate relation
                'reverseDirection': bool, // if relation is reversed from default direction
                'relationRatio': float, // (optional) relation parameter
                'relationLength': float, // (optional) relation parameter
                'mates': list[int] // indices into mate list of constrained mates
            }, ...
        ],
        'mateGroups': [
            {
                'name': str, // name of group -- may not be unique
                'id': str, // unique id of group in assembly
                'occurrences': list[int], // list of indices into occurrence list of grouped occurrences
                'has_parasolid': bool, // if all referenced parts are in parasolid.zip
                'has_step': bool // if all referenced parts are in in step.zip
            }, ...
        ],
        'subAssemblies': [
            {
                'id': str, // unique of subassembly relative to assembly
                'assemblyId': str // 
            }, ...
        ]

    }
    </pre>
</details>

<details>
<summary><strong>2-Metadata Tables</strong></summary>
Three parquet files contain metadata about parts, assemblies, and mates. All distances (and derived units) are
measured in meters. Masses are derived from assuming a unitless density of 1.  

`assemblies.parquet` has the following columns:
 - `assemblyId`: unique assembly id, used to find file in zip 'assemblies/{assemblyId}.zip' and in subAssembly lists
 - `n_subassemblies`: number of unique subassemblies (excluding the root) flattened into this assembly. Does not count multiple instances of the same subassembly.
 - `n_parts`: Number of unique parts in the assembly
 - `n_parasolid`: Number of parts for which we have parasolid geometry
 - `n_parasolid_errors`: Number of parts for which parasolid geometry has some error (see `parts.parquet` for error details)
 - `n_step`: Number of parts for which we have step geometry
 - `n_occurrences`: Number of part occurrences in the assembly
 - `n_mates`: Number of mates in the assembly
 - `n_ps_mates`: Number of mates in the assembly for which we have parasolid geometry for both parts
 - `n_step_mates`: Number of mates in the assembly for which we have step geometry for both parts
 - `n_groups`: Number of mate groups in the assembly
 - `n_relations`: Number of mate relations in the assembly
 - `is_subassembly`: If this assembly is included in any other assembly of the dataset as a subassembly.

`mates.parquet` has the following columns:
- `mateType`: Type of constaint the mate forms 
- `mcfs`: List of flattened 4x4 mating coordinate frame matrices for mate (see paper for details)
- `has_step`: If we have step geometry for both parts in the mate
- `parts`: List of part_ids for the two parts of the mate
- `ps_has_errors`: If either of the parasolid forms of the parts has an error (see `parts.parquet` for error details)

`parts.parquet` has the following columns. Note that summary and error data is computed for the parasolid files and may not be exactly the same for step versions if they exist:
 - `part_id`: unique id of part -- used to locate part in zip files
 - `readable`: if the parasolid file was readable
 - `n_parts`: number of parts in the parasolid file. Should always be 1
 - `n_bodies`: number of topological bodies in the parasolid file, should always be 1
 - `has_corrupt_state`: if the parasolid geometry has any corrupt state
 - `has_invalid_state`: if the parasolid geometry has any invalid state
 - `has_missing_geometry`: if any topological entities in the parasolid file has no associated geometry
 - `error_checking_topology`: if an error occurred reading any of the topological entities
 - `error_finding_bounding_box`: if an error occurred asking parasolid for the part's bounding box
 - `error_finding_na_box`: if an error occrured asking parasolid for the part's non-axis-aligned bounding box
 - `error_computing_mass_properties`: if an error occurred asking parasolid for the part's mass properties
 - `n_faults`: number of faults found by the parasolid kernel when processing the part
 - `n_faces_no_geo`: number of topological faces with missing geometry
 - `n_edges_no_geo`: number of topological edges with missing geometry
 - `n_verts_no_geo`: number of topological vertices with missing geometry
 - `n_topols`: number of topological entities
 - `n_relations`: number of relationships between topological entities
 - `n_regions`: number of topological regions
 - `n_shells`: number of topological shells
 - `n_faces`: number of topological faces
 - `n_edges`: number of topological edges
 - `n_loops`: number of topological loops
 - `n_vertices`: number of topological vertices
 - `n_plane`: number of face topologies with planar geometry
 - `n_cyl`: number of face topologies with cylindrical geometry
 - `n_cone`: number of face topologies with conical geometry
 - `n_sphere`: number of face topologies with spherical geometry
 - `n_torus`: number of face topologies with toroidal geometry
 - `n_bsurf`: number of face topologies with b-spline surface geometry
 - `n_offset`: number of face topologies with offset surface geometry
 - `n_fsurf`: number of face topologies with foreign (imported) surface geometry
 - `n_swept`: number of face topologies with swept surface geometry
 - `n_spun`: number of face topologies with spun surface geometry
 - `n_blendsf`: number of face topologies with blend-surface geometry
 - `n_line`: number of edge topologies with line geometry
 - `n_circle`: number of edge topologies with circular geometry
 - `n_ellipse`: number of edge topologies with elliptical geometry
 - `n_bcurve`: number of edge topologies with b-spline geometry
 - `n_icurve`: number of edge topologies with intersection curve geometry
 - `n_fcurve`: number of edge topologies with foriegn (imported) geometry
 - `n_spcurve`: number of edge topologies with surface parameterized geometry
 - `n_trcurve`: number of edge topologies with trimmed curve geometry
 - `n_cpcurve`: number of edge topologies with cpcurve geometry
 - `bb_0`: axis-aligned bounding box min corner x
 - `bb_1`: axis-aligned bounding box min corner y
 - `bb_2`: axis-aligned bounding box min corner z
 - `bb_3`: axis-aligned bounding box max corner x
 - `bb_4`: axis-aligned bounding box max corner y
 - `bb_5`: axis-aligned bounding box max corner z
 - `nabb_axis_0`: non-axis-aligned bounding box coordinate system z-axis x-coordinate
 - `nabb_axis_1`: non-axis-aligned bounding box coordinate system z-axis y-coordinate
 - `nabb_axis_2`: non-axis-aligned bounding box coordinate system z-axis z-coordinate
 - `nabb_loc_0`: non-axis-aligned bounding box coordinate system center x-coordinate
 - `nabb_loc_1`: non-axis-aligned bounding box coordinate system center y-coordinate
 - `nabb_loc_2`: non-axis-aligned bounding box coordinate system center z-coordinate
 - `nabb_ref_0`: non-axis-aligned bounding box coordinate system x-axis x-coordinate
 - `nabb_ref_1`: non-axis-aligned bounding box coordinate system x-axis y-coordinate
 - `nabb_ref_2`: non-axis-aligned bounding box coordinate system x-axis z-coordinate
 - `nabb_box_0`: non-axis-aligned bounding box min corner x
 - `nabb_box_1`: non-axis-aligned bounding box min corner y
 - `nabb_box_2`: non-axis-aligned bounding box min corner z
 - `nabb_box_3`: non-axis-aligned bounding box max corner x
 - `nabb_box_4`: non-axis-aligned bounding box max corner y
 - `nabb_box_5`: non-axis-aligned bounding box max corner z
 - `mp_amount`: total volume
 - `mp_mass`: total mass
 - `c_of_g_0`: center of gravity x-coordinate
 - `c_of_g_1`: center of gravity y-coordinate
 - `c_of_g_2`: center of gravity z-coordinate
 - `m_of_i_0`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_1`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_2`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_3`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_4`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_5`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_6`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_7`: moment of inertia tensor components relative to center of mass (row-major)
 - `m_of_i_8`: moment of inertia tensor components relative to center of mass (row-major)
 - `mp_periphery`: total surface area
 - `has_step`: if we have an associated step version
 - `uniqueid`: unique id used for deduplication (file should already be deduplicated)
 - `has_step_rep`: unused (leftover from deduplication)
 - `rep_part_id`: usused (leftover from deduplication)
 - `is_rep`: unused (leftover from deduplication)
 - `has_error`: if any of the error columns are true or non-zero.
</details>

<details>
<summary><strong>3-Associating with OnShape IDs</strong></summary>
All unique identifies are derived from their unique identifiers within Onshape. However, since Onshape query strings are case sensitive, contain non-path-friendly characters, and are too long for some file systems, we have canonicalized and shortened them. In general, the association is given by

 <pre>
 {documentId}_{documentMicroversion}_{elementId}_{encoded_configuration}_{[encoded_part_id if relevant]}
 </pre>

 associated files are named with this id plus the relevant file extension (.json, .x_t, or .step).

documentId, documentMicroversion, and elementId are unchanged from their onshape form except for being lower case only. The encoded configuration is the first 8 characters of the base32 encoded sha256 hash of the full configuration query string from Onshape. Because this is a destructive transform, the .json file `config_encodings.json` is provided to map back to the original, unencoded query strings. Part id is also encoded as a base32 encoding of the original Onshape part_id, but this transform is reversible. The file `file_encodings.py` contains helper functions for converting back-and-forth between Onshape identifiers and the identifiers used in the AutoMate dataset.
</details>

</br>

__Sample files:__  
json file: [00a0c68f4057bd42c6570577_c81c4b608085e9dcf0a9ee9a_3b1b46a08db95a944f3ba0f4.json](data/data_AutoMate/complete_top_level_assys_json/00a0c68f4057bd42c6570577_c81c4b608085e9dcf0a9ee9a_3b1b46a08db95a944f3ba0f4.json)  
x_t (parasolid text) file: [00a0c68f4057bd42c6570577_c81c4b608085e9dcf0a9ee9a_3ce90d8498cfab11d1e6008e_default_jjeei.x_t](data/data_AutoMate/complete_top_level_assys_parasolid/00a0c68f4057bd42c6570577_c81c4b608085e9dcf0a9ee9a_3ce90d8498cfab11d1e6008e_default_jjeei.x_t)  
step file: [00a0c68f4057bd42c6570577_c81c4b608085e9dcf0a9ee9a_3ce90d8498cfab11d1e6008e_default_jjeei.step](data/data_AutoMate/complete_top_level_assys_step/00a0c68f4057bd42c6570577_c81c4b608085e9dcf0a9ee9a_3ce90d8498cfab11d1e6008e_default_jjeei.step)  

Here, we provide tools to prepare the required data for training and testing of the model. The sample above is organized by using these scripts:
- [`grab_step_para_data_prep.ipynb`](./notebooks/grab_step_para_data_prep.ipynb) is used to extract the STEP and Parasolid files for the associated parts in the JSON files and save them in separate folders.
- [`json_convert_bprep_org.ipynb`](./notebooks/json_convert_bprep_org.ipynb) is used to convert the JSON files to a specific BREP organization that is required for the training and testing of the model.

## Installation
Automate relies on C++ extension modules to read Parasolid and STEP files. Since Parasolid is a proprietary CAD kernel, we can't distribute it, so you need to have the distribution on your machine already and compile it at install time.

### Requirements
Installation relies on CMake, OpenCascade, and Parasolid. In the future, we intend to make the Parasolid dependency optional. The easiest way to get the first two dependencies (and all python dependencies) is to install the conda environments `environment.yml` or `minimal_env.yml`:

`conda env create -f [environment|minimal_env].yml`

<details>
<summary>Parasolid</summary>

The Parasolid requirement relies on setting the environmental variable `$PARASOLID_BASE` on your system pointing to the Parasolid install directory for your operating system. For example

``export PARASOLID_BASE=${PATH_TO_PARASOLID_INSTALL}/intel_linux/base``

Replace ``intel_linux`` with the directory appropriate to your OS. The base directory should contain files like `pskernel_archive.lib` and `parasolid_kernel.h`.

Once these requirements are met, you an install via pip:

`pip install git+https://github.com/degravity/automate.git@v1.0.4`
</details>

<details>
<summary>Added Installation on AMD-based System (Craftnetics TensorCraft - AMD-based Workstation)</summary>

1- Compatible docker container from DockerHub is [rocm/pytorch:rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.10.0](https://hub.docker.com/layers/rocm/pytorch/rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.10.0/images/sha256-34313368f1563d92e5fd49837a705df5ad85d6d6ee466330d3bb17b6b78ac100?context=explore) that one can pull via:
```bash
$ docker pull rocm/pytorch:rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.10.0
```
2- Run the container via (the container name will be `automate`):
```bash
$ sudo docker run -it --name automate --cap-add=SYS_PTRACE \
                  --security-opt seccomp=unconfined \
                  --device=/dev/kfd --device=/dev/dri \
                  --group-add $(getent group video | cut -d':' -f 3) \
                  --ipc=host --network=host --dns 8.8.8.8 \
                  -v /path/to/volume:/path/to/volume \
                  -p 0.0.0.0:6005:6005 -e DISPLAY=$DISPLAY \
                  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
                  rocm/pytorch:rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.10.0
```
3- The docker container above comes with required PyTorch installation (1.10.0) that is compatible with ROCm stack (Please check via `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`).  
Below are the steps to install the remaining dependencies:
```bash
$ source activate base                  # To activate the base environment of the container,
$ conda install mamba -c conda-forge --strict-channel-priority --override-channels -y # To install and use mamba as the package manager
$ pip install pytorch-lightning==1.7.3  # To install pytorch-lightning
$ conda install matplotlib -y
$ conda install -c conda-forge dotmap -y
$ conda install -c conda-forge eigen -y
$ conda install cmake -y
$ conda install -c conda-forge pybind11 -y
$ conda install seaborn -y
$ conda install -c conda-forge occt=7.6 --strict-channel-priority --override-channels -y # occt 7.6 or 7.7 should work 
```
__Notes:__  
[1] In addition to the dependencies above, we need to install `xxhash` and `setuptools` via pip as:
```bash
$ pip install xxhash setuptools==59.5.0 # setuptools is solving "AttributeError: module 'distutils' has no attribute 'version' issue"
```
[2] All of these can be moved to an environment file. However, we would like to explicitly present the dependencies and installation steps to avoid potential issues.  

4-`torch_geometric` is a bit complicated to install on AMD-based systems. There are some attempts to share [some binaries](https://github.com/Looong01/pyg-rocm-build/) to install and also as suggested in the documentation of the [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#:~:text=number%20here.-,ROCm,-%3A%20The%20external). However, it might be still complicated to install due to the differences between the compiled system that is released on the repository and the users' systems (e.g., see [here](https://github.com/Looong01/pyg-rocm-build/issues/3)). Regardless, one can compile and install the individual dependencies via (assuming we are inside different folder ---i.e, ./pyg--- then the cloned repository):  

torch-scatter:
```bash
$ git clone https://github.com/rusty1s/pytorch_scatter.git
$ cd pytorch_scatter
$ git checkout 2.0.9
$ pip . install -vvv |& tee build_scatter.log # After |& is to log the installation process
```

torch-sparse:
```bash
$ git clone https://github.com/rusty1s/pytorch_sparse.git
$ cd pytorch_sparse
$ git checkout 0.6.13
$ pip . install -vvv |& tee build_sparse.log # After |& is to log the installation process
```

torch-spline-conv:
```bash
$ git clone https://github.com/rusty1s/pytorch_spline_conv
$ cd pytorch_spline_conv
$ git checkout 2.0.3
$ pip . install -vvv |& tee build_spline_conv.log # After |& is to log the installation process
```

torch-cluster (optional):  
```bash
$ git clone https://github.com/rusty1s/pytorch_cluster.git
$ cd pytorch_cluster
$ git checkout 1.2.1
$ pip . install -vvv |& tee build_cluster.log # After |& is to log the installation process
```
__Note:__ For C++ implementation of torch-cluster, please check this [link](https://github.com/rusty1s/pytorch_cluster#c-api).  

torch-geometric:  
```bash
$ git clone https://github.com/pyg-team/pytorch_geometric.git
$ cd pytorch_geometric
$ git checkout 2.0.3
$ pip . install -vvv |& tee build_geometric.log
```

__Notes:__  
[1] `mamba` package manager might work relatively faster than `conda`. After the `conda update conda` above, one can install `mamba`.  
[2] In some cases, even the installation of the mamba would be problematic. Therefore, one can use a logic similar to:
`conda install mamba -c conda-forge --strict-channel-priority --override-channels -y` to limit the number of channels to only `conda-forge`.  
[3] It is important to pay attention to the version of the PyTorch and its compatibility to the specific PyTorch Geometric stack---and its dependencies as provided on step 4 above.  
[4] We can also create Python binaries (e.g., `.whl` files) for each of the torch-geometric and other libraries above. This will avoid the recompilation in the future. We can `python setup.py bdist_wheel`. This command needs to be run in the main folder of the target library (e.g., ./pyg/pytorch_geometric/) where `setup.py` is located. For example, the compiled binary of `torch_geometric` will be in `./pytorch_geometric/dist/`.
</details>

### Training of the Model (wip)
One can use `train.py` to train the model. The model is trained using the Mean Squared Error (MSE) loss between the ground truth points and the reconstructed points. Below is an example command to run the training with single GPU.
```bash
$ python train.py --splits_path ./data/data_AutoMate/complete_top_level_assys_json \
                  --data_dir ./data/data_AutoMate/complete_top_level_assys_step \
                  --gpus 1 --max_epochs 100
```

### Troubleshooting
```
ImportError: dynamic module does not define module export function (PyInit_automate_cpp)
```

If you get this error when trying to import part of the module, it means that python can't find the C++ extensions module. To fix this, try cleaning out **all** build files and building again.

## Citing
If you use this code our the [AutoMate Dataset](https://zenodo.org/record/7776208#.ZDcYinbMIQ8) (or any derivatives e.g., [complete_top_level_assemblies](https://drive.google.com/file/d/100wKGZjeAt0fw0hVG_D0vLpDho0_zprd/view)) in your work, please cite us:

```
@article{10.1145/3478513.3480562,
author = {Jones, Benjamin and Hildreth, Dalton and Chen, Duowen and Baran, Ilya and Kim, Vladimir G. and Schulz, Adriana},
title = {AutoMate: A Dataset and Learning Approach for Automatic Mating of CAD Assemblies},
year = {2021},
issue_date = {December 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {40},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3478513.3480562},
doi = {10.1145/3478513.3480562},
month = {dec},
articleno = {227},
numpages = {18},
keywords = {assembly-based modeling, representation learning, boundary representation, computer-aided design}
}
```
