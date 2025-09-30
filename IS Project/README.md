# 🧠 Brain Tumor MRI Image Classification & Predictive Generation

## 📌 Project Overview
This project is developed as part of our school group assignment.  
The system provides two key functionalities:

1. **Tumor Classification**  
   - Upload an MRI image and classify whether a tumor is present.  
   - Implemented with a CNN (currently dummy random output).  

2. **Predictive Image Generation**  
   - Upload a pre-op MRI and the first post-op MRI to generate a predicted second post-op MRI.  
   - Implemented with a Pix2Pix cGAN (currently dummy image placeholder).  

The architecture integrates:
- **FastAPI (backend)** – provides APIs for classification and image generation.  
- **Streamlit (frontend)** – interactive UI with two separate pages for each function.  
- **Docker Compose** – containerized setup for one-click deployment.  

---

## 🏗️ Project Structure

```
project-root/
│── backend/                
│   ├── app/
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── models/
│   │   │   ├── classifier.py        # Dummy tumor classifier (random)
│   │   │   ├── generator.py         # Dummy image generator (PIL)
│   │   ├── utils/
│   │   │   ├── preprocess.py
│   │   │   ├── evaluate.py
│   ├── requirements.txt
│   └── Dockerfile
│
│── frontend/
│   ├── app/
│   │   ├── app.py                        # Landing page
│   │   ├── pages/
│   │   │   ├── 1_Tumor_Classification.py
│   │   │   ├── 2_Predictive_Image_Generation.py
│   ├── requirements.txt
│   └── Dockerfile
│
│── docker-compose.yml
│── .env.example
│── README.md
```

---

## ⚙️ Tech Stack
- **Python 3.11** (backend & frontend)
- **FastAPI** (backend API service)
- **Streamlit** (frontend UI)
- **Docker Compose** (orchestration)
- **Pillow** (dummy image generation)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-repo>.git
cd <your-repo>
```

### 2. Setup Environment Variables
Copy `.env.example` to `.env` (optional for later secrets like DB/API keys):
```bash
cp .env.example .env
```

### 3. Build & Run with Docker Compose
```bash
docker-compose up --build
```

### 4. Access Services
- **Backend API** → [http://localhost:8000](http://localhost:8000)  
- **Streamlit Frontend** → [http://localhost:8501](http://localhost:8501)  

---

## 🖥️ Usage

### Tumor Classification
- Navigate to **"Tumor Classification"** page in Streamlit.  
- Upload an MRI image.  
- The backend (dummy model) will return `True` or `False`.  

### Predictive MRI Generation
- Navigate to **"Predictive Image Generation"** page.  
- Upload **Pre-op MRI** and **1st Post-op MRI**.  
- The backend (dummy model) will generate a placeholder MRI image.  

---

## 🧑‍🤝‍🧑 Team Collaboration
- **Group A** → works on `pages/1_Tumor_Classification.py` + `backend/app/models/classifier.py`  
- **Group B** → works on `pages/2_Predictive_Image_Generation.py` + `backend/app/models/generator.py`  

Best practice:
- Use feature branches (`feature/classifier`, `feature/generator`)  
- Submit Pull Requests for review  
- Keep `main` branch stable for Docker deployment  

---

## 🧪 Testing API (Optional)
After containers are up, test APIs with `curl`:

```bash
# Tumor Classification
curl -X POST "http://localhost:8000/classify"   -F "file=@sample_mri.jpg"

# Predictive Image Generation
curl -X POST "http://localhost:8000/generate"   -F "file1=@pre_op.jpg"   -F "file2=@post_op1.jpg"
```

---

## 📊 Future Enhancements
- Replace dummy models with real **CNN classifier** (PyTorch/TensorFlow).  
- Implement **Pix2Pix / GAN** for MRI predictive image generation.  
- Add **unit tests** (`pytest`) for backend endpoints.  
- Deploy on **cloud (AWS/GCP/Azure/Heroku/Render)**.  

---

## 📚 References
- Fan et al. (2022). *CACA guidelines for holistic integrative management of glioma*.  
  Holistic Integrative Oncology, 1(1), 22.  
  [https://doi.org/10.1007/s44178-022-00020-x](https://doi.org/10.1007/s44178-022-00020-x)  

---
