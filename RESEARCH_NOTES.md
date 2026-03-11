# Research: 3D Mesh Segmentation, Ontologies, and Test Data Sources

## 1. 3D Mesh Segmentation Methods for CAD Reconstruction

### 1.1 Approximate Convex Decomposition (ACD)

Exact convex decomposition is NP-hard. Approximate methods relax the convexity constraint to produce "nearly convex" parts with concavity below a user-defined threshold.

**Key methods:**

| Method | Year | Key Idea | Notes |
|--------|------|----------|-------|
| **V-HACD** (Mamou) | 2009+ | Hierarchical dual-graph decimation with concavity cost function | Now **deprecated/archived**. Header-only C++ (v4.0). [GitHub](https://github.com/kmammou/v-hacd) |
| **CoACD** (Wei et al.) | 2022 | Collision-aware concavity metric + tree search for cutting planes | **Recommended successor to V-HACD**. Cuts meshes with 3D planes directly (no voxelization). [Project page](https://colin97.github.io/CoACD/), [ACM TOG](https://dl.acm.org/doi/abs/10.1145/3528223.3530103) |
| **CoRiSe** | 2016 | Convex Ridge Separation — identifies protruding parts via residual concavity | Single parameter: concavity tolerance. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448516300355) |

**Why it matters for CAD:** Convex parts map well to extrusion/Boolean-subtract operations. Each nearly-convex piece can often be represented as a single extruded or revolved profile.

### 1.2 Shape Diameter Function (SDF) Based Segmentation

The SDF is a scalar function on a mesh surface measuring **local object thickness/diameter**. It distinguishes thick vs. thin parts volumetrically.

**Algorithm pipeline:**
1. **SDF computation** — cast rays from inverted cone at each face, average penetration distances
2. **Soft clustering** — fit k Gaussian distributions (GMM) to SDF value distribution, initialized with k-means++
3. **Hard clustering (graph-cut)** — refine using dihedral angle and concavity as edge weights

**Key properties:**
- **Pose-invariant** — similar values for analogous parts across poses
- **Consistent** — produces similar segmentations across object families

**Implementations:**
- **CGAL** `Surface_mesh_segmentation` package — production-ready C++ implementation. [CGAL docs](https://doc.cgal.org/latest/Surface_mesh_segmentation/index.html)
- **Fast/Robust SDF** (Chen 2018) — offset-surface approach, single ray per point, works on non-watertight meshes. [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190666)
- **Neural SDF** (Roy 2023) — deep learning to predict SDF values, then graph-cut. Orders of magnitude faster. [arXiv:2306.11737](https://arxiv.org/abs/2306.11737)

**Why it matters for CAD:** SDF naturally separates parts by thickness — a table's legs vs. top, a bottle's neck vs. body. These thickness-based segments often correspond to distinct CAD operations.

### 1.3 Skeletal / Medial Axis Based Segmentation

The Medial Axis Transform (MAT) encodes shape as a set of maximally inscribed spheres, capturing both geometry and topology.

**Key methods:**

| Method | Year | Key Idea |
|--------|------|----------|
| **SEG-MAT** | 2020 | Uses MAT to identify junction types between parts. 10x faster than prior methods. [GitHub](https://github.com/clinplayer/SEG-MAT), [IEEE TVCG](https://dl.acm.org/doi/abs/10.1109/TVCG.2020.3032566) |
| **Medial Skeletal Diagram** | 2024 | Generalized enveloping primitives for compact MAT. Used for mesh decomposition, alignment, compression. [ACM TOG](https://dl.acm.org/doi/10.1145/3687964), [Project](https://gmh14.github.io/medial-skeletal-diagram/) |
| **Coverage Axis++** | 2024 | Set-cover formulation — cover surface with fewest medial balls. Global structure awareness. [Wiley CGF](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15143) |
| **Point2Skeleton** | 2021 | Unsupervised learning of skeletal representations from point clouds. Works on non-watertight/non-tubular shapes. |

**Why it matters for CAD:** The skeleton reveals sweep paths (for loft/sweep operations) and rotational axes (for revolve operations). Junction detection identifies where to split extrude-from-revolve transitions.

### 1.4 Normal-Based Clustering / Region Growing

Groups adjacent faces with similar surface normals into coherent regions.

**Algorithm:**
1. Sort faces by curvature (start from flattest)
2. Grow region by adding neighbors whose normal angle difference < threshold
3. Use curvature threshold to add new seed points
4. Post-process: merge small regions to reduce over-segmentation

**Key parameters:**
- **Smoothness threshold** — max normal deviation angle (radians)
- **Curvature threshold** — max curvature disparity for seed expansion
- **Min/max cluster size** — discard too-small or too-large clusters

**Implementations:**
- **PCL (Point Cloud Library)** — `RegionGrowing` class. [PCL docs](https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html)
- **Open3D** — normal-based clustering utilities
- **CGAL** — via custom functors on `Surface_mesh`

**Extensions:**
- Curvedness-based region growing (Lavoué et al.) — uses curvedness instead of raw normals
- HMRF-EM segmentation — Hidden Markov Random Field + EM for unsupervised mesh segmentation

**Why it matters for CAD:** Flat/cylindrical/spherical regions with consistent normals map directly to planar faces, cylindrical surfaces, and fillets in CAD models. The simplest and most robust first-pass segmentation.

### 1.5 Hierarchical Segmentation

Produces a tree of segments from coarse to fine decomposition.

**Key methods:**

| Method | Year | Key Idea |
|--------|------|----------|
| **Fuzzy Clustering and Cuts** (Katz & Tal) | 2003 | Hierarchical decomposition via fuzzy clustering. ACM TOG. |
| **Randomized Cuts** (Golovinskiy & Funkhouser) | 2008 | Top-down binary splits via randomized minimum cuts. |
| **Topology-Driven** (Tierny et al.) | 2007 | Uses topological features (Reeb graph) for hierarchy. |
| **Hierarchical Splat Clustering** (Zhang et al.) | 2015 | Local similarity metric in hierarchical clustering framework. |
| **Spectral Clustering + Negative Curvature** | 2017 | Affinity matrix from min-curvature + dihedral angle. Fully unsupervised. |
| **SHRED** | 2022 | Learned local operations with merge-threshold for granularity control. [ACM TOG](https://dl.acm.org/doi/10.1145/3550454.3555440) |
| **PartField** | 2025 | Feedforward 3D feature field learning. 20% more accurate, orders of magnitude faster. [arXiv](https://arxiv.org/html/2504.11451v1) |

**Why it matters for CAD:** Hierarchical segmentation lets you choose decomposition granularity — coarse for major body/limb separation, fine for individual fillets and chamfers.

### 1.6 Sweep / Profile Detection for CAD Reconstruction

The reverse-engineering pipeline for converting meshes to CAD operations:

**General workflow:**
1. **Segment** mesh into regions
2. **Cut** mesh with planes to extract 2D cross-section profiles
3. **Fit** parametric curves (arcs, lines, splines) to cross-sections
4. **Identify operation type**: extrude (constant profile), revolve (profile + axis), loft (varying profiles), sweep (profile + path)
5. **Boolean combine** the resulting solids

**Academic approaches:**
- **Benko & Varady** — Direct Segmentation Method: fit hierarchy of surfaces (planes → cylinders → extrusion surfaces → revolution surfaces)
- **Interactive Reverse Engineering** (2024) — plane-cut → fit edges → extrude/loft/revolve/sweep → Boolean merge. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167839624000736)
- **SECAD-Net / Free2CAD** — deep learning: predict 2D sketches, extrude to 3D, Boolean combine

**Commercial tools implementing this:**
- **PolyWorks|Modeler** — extract sketch planes, fit parametric entities
- **QUICKSURFACE** — mark features (planes, cylinders, revolved/extruded surfaces), section extraction
- **Mesh2Surface** (SOLIDWORKS plugin) — mesh-to-CAD with 3D comparison
- **EXModel (SHINING 3D)** — interactive cutting + sketch extraction

---

## 2. 3D Object Part Ontologies and Taxonomies

### 2.1 ShapeNet
- **Scale:** 3,000,000+ indexed models, 220,000 classified into 3,135 WordNet synset categories
- **ShapeNetCore:** 55 common categories, ~51,300 models with verified alignment
- **ShapeNetSem:** 270 categories, 12,000 models with real-world dimensions + materials
- **Annotations:** rigid alignments, parts, bilateral symmetry planes, physical sizes, keywords
- **Taxonomy:** organized under WordNet hierarchy
- [shapenet.org](https://shapenet.org/)

### 2.2 PartNet
- **Scale:** ~26,000 shapes across 24 categories with fine-grained hierarchical part annotations
- **Part hierarchy:** 3 levels of semantic granularity (level-1, level-2, level-3)
- **Tasks:** fine-grained semantic segmentation, hierarchical segmentation, instance segmentation
- **Applications:** shape analysis, dynamic 3D scene modeling, simulation, affordance analysis
- [partnet.cs.stanford.edu](https://partnet.cs.stanford.edu/)
- [GitHub](https://github.com/daerduoCarey/partnet_dataset)

### 2.3 PartNeXt (2025)
- **Improvement over PartNet:** 50 categories (vs. 24), annotates directly on textured meshes
- Novel web-based annotation interface
- [arXiv](https://arxiv.org/html/2510.20155v1)

### 2.4 Princeton Shape Benchmark (PSB)
- **Scale:** 1,814 models (v1), 6,670 unique models total
- **Multiple classification schemes:** by function, by function+form, by construction (man-made vs. natural)
- **Format:** .off files + metadata + thumbnails
- [shape.cs.princeton.edu/benchmark](https://shape.cs.princeton.edu/benchmark/)

### 2.5 CAD-Specific Feature Ontologies
- **STEP AP242 + OWL Knowledge Graph** — automatic feature recognition from STEP files, stores geometric/topological features in ontology. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558932/)
- **Unified Shape Feature Taxonomy** (Gupta & Gurumoorthy) — three classes: volumetric features, deformation features, free-form surface features
- **Form-to-Semantic Feature Ontology** — OWL-based, maps form features to manufacturing semantics. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1474034620300574)
- **ALIGN-Parts** — unified ontology across PartNet, 3DCoMPaT++, Find3D: 1,794 unique 3D parts

### 2.6 Evaluation Benchmarks
- **Princeton Segmentation Benchmark** — ground-truth segmentations for evaluating mesh segmentation. [cs.princeton.edu](https://www.cs.princeton.edu/~funk/segeval.pdf)
- **Metrics:** 3D Normalized Probabilistic Rand Index (3DNPRI), Weighted Levenshtein Distance, Adaptive Entropy Increment

---

## 3. CIFAR-100 Complete Category List

60,000 32x32 color images. 100 fine classes grouped into 20 superclasses. 500 train + 100 test images per class.

Source: [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

| # | Superclass | Fine Classes |
|---|------------|-------------|
| 1 | **Aquatic mammals** | beaver, dolphin, otter, seal, whale |
| 2 | **Fish** | aquarium fish, flatfish, ray, shark, trout |
| 3 | **Flowers** | orchids, poppies, roses, sunflowers, tulips |
| 4 | **Food containers** | bottles, bowls, cans, cups, plates |
| 5 | **Fruit and vegetables** | apples, mushrooms, oranges, pears, sweet peppers |
| 6 | **Household electrical devices** | clock, computer keyboard, lamp, telephone, television |
| 7 | **Household furniture** | bed, chair, couch, table, wardrobe |
| 8 | **Insects** | bee, beetle, butterfly, caterpillar, cockroach |
| 9 | **Large carnivores** | bear, leopard, lion, tiger, wolf |
| 10 | **Large man-made outdoor things** | bridge, castle, house, road, skyscraper |
| 11 | **Large natural outdoor scenes** | cloud, forest, mountain, plain, sea |
| 12 | **Large omnivores and herbivores** | camel, cattle, chimpanzee, elephant, kangaroo |
| 13 | **Medium-sized mammals** | fox, porcupine, possum, raccoon, skunk |
| 14 | **Non-insect invertebrates** | crab, lobster, snail, spider, worm |
| 15 | **People** | baby, boy, girl, man, woman |
| 16 | **Reptiles** | crocodile, dinosaur, lizard, snake, turtle |
| 17 | **Small mammals** | hamster, mouse, rabbit, shrew, squirrel |
| 18 | **Trees** | maple, oak, palm, pine, willow |
| 19 | **Vehicles 1** | bicycle, bus, motorcycle, pickup truck, train |
| 20 | **Vehicles 2** | lawn-mower, rocket, streetcar, tank, tractor |

---

## 4. Free STL Model Sources for Testing

### 4.1 Dead Trees / Bare Trees (Branching Geometry)

| Source | Model | URL |
|--------|-------|-----|
| Thingiverse | Dead Tree by warmarine759 | [thing:3839198](https://www.thingiverse.com/thing:3839198) |
| Thingiverse | Customizable Tree (parametric, ~12 params) | [thing:279864](https://www.thingiverse.com/thing:279864) |
| Thingiverse | All "dead_tree" tagged models | [thingiverse.com/tag:dead_tree](https://www.thingiverse.com/tag:dead_tree) |
| Printables | Dead Tree (Supportless) - Tabletop Terrain | [printables.com/model/126352](https://www.printables.com/model/126352-dead-tree-supportless-tabletop-terrain) |
| Printables | Tree Branch / Shrub Armature | [printables.com/model/100479](https://www.printables.com/model/100479-tree-branch-shrub-armature-dd-scatter) |
| Cults3D | Dead Tree | [cults3d.com](https://cults3d.com/en/3d-model/game/dead-tree) |
| Free3D | 81 free tree STL models | [free3d.com/3d-models/stl-tree](https://free3d.com/3d-models/stl-tree) |
| TurboSquid | 30+ free tree STLs | [turbosquid.com](https://www.turbosquid.com/3d-model/free/trees/stl) |

### 4.2 Humanoid Figures

| Source | URL |
|--------|-----|
| Thingiverse | [thingiverse.com/tag:human](https://www.thingiverse.com/tag:human) |
| GrabCAD | [grabcad.com/library?query=human+figure](https://grabcad.com/library?per_page=20&query=human+figure) |
| Sketchfab | [sketchfab.com/tags/human](https://sketchfab.com/tags/human) |
| Cults3D | [cults3d.com/en/tags/human](https://cults3d.com/en/tags/human?only_free=true) (1,600+ free) |
| STLFinder | [stlfinder.com/3dmodels/human-figure](https://www.stlfinder.com/3dmodels/human-figure/) |

### 4.3 General Everyday Objects

| Platform | Notes | URL |
|----------|-------|-----|
| **Thingiverse** | Largest free repo. Millions of models. | [thingiverse.com](https://www.thingiverse.com/) |
| **Printables** | #1 Thingiverse alternative, all free | [printables.com](https://www.printables.com/) |
| **GrabCAD** | 4.5M+ CAD files, professional grade, 8M+ engineers | [grabcad.com/library](https://grabcad.com/library) |
| **MyMiniFactory** | Curated, tested for printability | [myminifactory.com](https://www.myminifactory.com/) |
| **Cults3D** | Both free and paid, well-organized | [cults3d.com](https://cults3d.com/en) |
| **Sketchfab** | View in browser, many downloadable | [sketchfab.com](https://sketchfab.com/) |
| **TurboSquid** | Free section available, professional quality | [turbosquid.com](https://www.turbosquid.com/) |
| **Free3D** | Dedicated free section | [free3d.com](https://free3d.com/) |
| **STLFinder** | Aggregator across all platforms | [stlfinder.com](https://www.stlfinder.com/) |
| **Yeggi** | Search engine for 3D printable models | [yeggi.com](https://www.yeggi.com/) |

---

## 5. Open-Source Libraries Summary

| Library | Language | Key Feature | URL |
|---------|----------|-------------|-----|
| **CGAL** | C++ (Python/Java bindings) | SDF segmentation, convex hulls, mesh processing | [cgal.org](https://www.cgal.org/) |
| **CoACD** | C++ | Modern approximate convex decomposition (replaces V-HACD) | [GitHub](https://colin97.github.io/CoACD/) |
| **V-HACD v4** | C++ header-only | Legacy ACD (archived, use CoACD instead) | [GitHub](https://github.com/kmammou/v-hacd) |
| **SEG-MAT** | C++ | Medial axis based segmentation | [GitHub](https://github.com/clinplayer/SEG-MAT) |
| **PCL** | C++ | Region growing, normal estimation | [pointclouds.org](https://pcl.readthedocs.io/) |
| **Open3D** | C++/Python | General mesh/point cloud processing | [open3d.org](http://www.open3d.org/) |
| **trimesh** | Python | Mesh loading, convex decomposition (wraps V-HACD) | [GitHub](https://github.com/mikedh/trimesh) |
