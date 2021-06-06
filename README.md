# UCSGNet: Unsupervised Discovering of Constructive Solid Geometry Tree
Reproducing Published Papers  
UCSGNet: Unsupervised Discovering of Constructive Solid Geometry Tree [PaperLink](https://arxiv.org/abs/2006.09102)  

Used author codes: Data loader, visualizer

# Reproducing the results
## Download Dataset
We use 2D CAD same as [CSGNet](https://github.com/hippogriff/CSGNet) and 3D ShapeNet same as [IM-NET](https://github.com/czq142857/IM-NET).All the data can be downloaded [here]().  

## Data 
All the data file should be placed same as the below.  
- data/
  - cad/
    - cad.h5
  - hdf5/
    - all_vox256_img_test.hdf5
    - all_vox256_img_test.txt
    - all_vox256_img_train.hdf5
    - all_vox256_img_train.txt
    - all_vox256_img_z.hdf5
  - pointcloud_surface/. . .
    - 02691156/ . . .


TimeLine:  
5/5 ∼ 5/19: Implement data generator, encoder, decoder, shapeeval, converter   
5/19 ∼ 5/26: Implement rest of the model, utility functions(i.e. loss, gumbelsoftmax, etc)  
5/26 ~ 6/3: Establish whole pipline and test, visualize them  
6/3 ∼ 6/6: Make reports, poster, organize codes  
