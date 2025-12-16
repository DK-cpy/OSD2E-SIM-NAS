# OSD2E-SIM-NAS
**中文版本**  
1. **Diffevo for GAN 文件夹**（对应 GAN 架构搜索任务）  
   - `diffevo-search_gen_arch.py`：搜索阶段的代码  
   - `diffevo-fully_train_arch.py`：重训练阶段的代码  
   - 以 `STL10` 开头的文件为 STL-10 数据集相关代码  

2. **requirements.txt 文件**  
   - 记录所需的库函数及其对应版本  

3. **onestep-DiffEvo-NAS 文件夹**（对应图像分类任务）  
   - 以 `onestep` 开头的三个文件分别为以下数据集的搜索代码：  
     - CIFAR-10  
     - CIFAR-100  
     - NAS-Bench-201  
   - `train_search.py`：重训练代码  

---

**英文版本**  
1. **The `Diffevo for GAN` folder** (for GAN architecture search tasks)  
   - `diffevo-search_gen_arch.py`: code for the search phase  
   - `diffevo-fully_train_arch.py`: code for the re-training phase  
   - Files prefixed with `STL10` are for the STL-10 dataset  

2. **The `requirements.txt` file**  
   - Lists the required libraries and their corresponding versions  

3. **The `onestep-DiffEvo-NAS` folder** (for image classification tasks)  
   - The three files starting with `onestep` are the search-phase codes for:  
     - CIFAR-10  
     - CIFAR-100  
     - NAS-Bench-201  
   - `train_search.py`: re-training code
