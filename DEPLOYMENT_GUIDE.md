# 🚀 Deployment & Testing Guide

## Quick Launch Options

### Option 1: Batch Files (Windows - EASIEST)

Double-click one of these files:
- `run_enhanced.bat` - ⭐ **Recommended** (Enhanced version with 20 layers)
- `run_mvp.bat` - MVP version (professional features)
- `run_basic.bat` - Basic version (lightweight)

The app will open in your default browser at `http://localhost:8501`

---

### Option 2: Command Line

```bash
# Make sure you're in the Mandala directory
cd c:\Users\ADMIN\Desktop\Mandala\Mandala

# Run Enhanced version (RECOMMENDED)
python -m streamlit run app_enhanced.py

# OR run MVP version
python -m streamlit run app_mvp.py

# OR run Basic version
python -m streamlit run app.py
```

---

## Installation Checklist

### ✅ Prerequisites
- [x] Python 3.9+ installed (You have Python 3.13.3)
- [x] pip package manager
- [x] All dependencies installed

### ✅ Dependencies Status
All packages have been successfully installed:
- streamlit 1.52.2
- opencv-python 4.12.0.88
- numpy 2.2.6
- Pillow 12.0.0
- svgwrite 1.4.3
- pandas 2.3.3

---

## Testing Checklist

### 🧪 Basic Functionality Tests

#### Test 1: Photo Engraving (Dithering)
1. Launch app: `python -m streamlit run app_enhanced.py`
2. Upload a test image (portrait photo recommended)
3. Go to "Photo Engraving" tab
4. Select "Floyd-Steinberg (Recommended)"
5. Click "Download Dithered PNG"
6. **Expected:** PNG file downloads successfully

#### Test 2: Vector Scoring
1. Upload a simple logo or icon
2. Go to "Vector Scoring" tab
3. Use default thresholds (50/150)
4. Adjust smoothness slider to 0.005
5. Click "Download SVG"
6. **Expected:** SVG file downloads, opens in Inkscape/LightBurn

#### Test 3: Multi-Layer (20 layers)
1. Upload a mandala or circular design
2. Go to "Multi-Layer" tab
3. Use material calculator:
   - Select "3mm Plywood"
   - Set 5 layers
4. Set 5 layers in main settings
5. Click "Generate Layers"
6. Click "Download All Layers (ZIP)"
7. **Expected:** ZIP with 5 SVG files downloads

#### Test 4: Beginner Mode
1. Check "Beginner Mode" in sidebar
2. Verify extra help text appears
3. Read the beginner guide (top of page)
4. **Expected:** Additional tooltips and warnings visible

#### Test 5: Measurement Tools
1. Enable "Add Measurement Overlay to Previews" in sidebar
2. Upload any image
3. Go to any tab
4. **Expected:** Rulers visible on preview images

---

## Troubleshooting

### Issue: "No module named streamlit"

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: App won't start

**Solution:**
```bash
# Check Python version
python --version

# Should be 3.9 or higher
# If not, install Python 3.12 or 3.13
```

### Issue: Import errors for cv2, PIL, etc.

**Solution:**
```bash
# Install specific package
pip install opencv-python
pip install Pillow
pip install numpy
```

### Issue: Batch file doesn't work

**Solution:**
```bash
# Run directly from command line
cd c:\Users\ADMIN\Desktop\Mandala\Mandala
python -m streamlit run app_enhanced.py
```

### Issue: Browser doesn't open automatically

**Solution:**
- Manually open browser
- Navigate to: `http://localhost:8501`
- Or try: `http://127.0.0.1:8501`

---

## Performance Notes

### File Processing Times (Approximate)

**1000x1000 pixel image:**
- Photo Engraving: 2-5 seconds
- Vector Scoring: 1-3 seconds
- Multi-Layer (5 layers): 3-8 seconds
- Multi-Layer (20 layers): 10-25 seconds

**2000x2000 pixel image:**
- Photo Engraving: 8-15 seconds
- Vector Scoring: 3-6 seconds
- Multi-Layer (5 layers): 8-15 seconds
- Multi-Layer (20 layers): 30-60 seconds

**Optimization Tips:**
1. Enable image resizing in sidebar (target width: 800-1200px)
2. Use lower layer counts (4-6) for faster processing
3. Close other browser tabs for better performance

---

## Network Access (Optional)

To allow others on your network to access:

```bash
# Run with network address
python -m streamlit run app_enhanced.py --server.address=0.0.0.0

# Then share your IP address:
# Others can access at: http://YOUR_IP_ADDRESS:8501
```

**Security Note:** Only do this on trusted networks!

---

## Production Deployment (Advanced)

### Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repository
4. Select `app_enhanced.py` as main file
5. Deploy!

### Deploy to Heroku

1. Create `Procfile`:
```
web: streamlit run app_enhanced.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `runtime.txt`:
```
python-3.12.0
```

3. Deploy:
```bash
git init
heroku create
git push heroku main
```

---

## File Structure Check

Your directory should look like this:

```
Mandala/
│
├── app.py                      ✅ Basic version
├── app_mvp.py                  ✅ MVP version
├── app_enhanced.py             ✅ Enhanced version
├── requirements.txt            ✅ Dependencies
│
├── run_enhanced.bat            ✅ Windows launcher (enhanced)
├── run_mvp.bat                 ✅ Windows launcher (MVP)
├── run_basic.bat               ✅ Windows launcher (basic)
│
├── README.md                   ✅ Main documentation
├── QUICK_START.md              ✅ Quick start guide
├── MVP_IMPROVEMENTS.md         ✅ MVP research
├── ENHANCED_FEATURES.md        ✅ Enhanced features
├── VERSION_COMPARISON.md       ✅ Version comparison
└── DEPLOYMENT_GUIDE.md         ✅ This file
```

---

## Browser Compatibility

**Recommended Browsers:**
- ✅ Google Chrome (best performance)
- ✅ Microsoft Edge (Chromium)
- ✅ Firefox
- ⚠️ Safari (may have issues with file downloads)

---

## System Requirements

### Minimum:
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- Disk: 500 MB free space
- OS: Windows 10, macOS 10.14+, Linux

### Recommended:
- CPU: Quad-core 2.5 GHz or higher
- RAM: 8 GB or more
- Disk: 1 GB free space
- SSD for faster image processing

---

## First Time Setup (Complete)

1. ✅ Install Python 3.9+
2. ✅ Install dependencies: `pip install -r requirements.txt`
3. ✅ Test basic version: Double-click `run_basic.bat`
4. ✅ Test MVP version: Double-click `run_mvp.bat`
5. ✅ Test enhanced version: Double-click `run_enhanced.bat`
6. Upload a test image to each
7. Try all three tabs
8. Download outputs and verify they work in LightBurn

---

## Next Steps After Testing

1. **Choose your preferred version:**
   - Beginners → `app_enhanced.py`
   - Advanced users → `app_enhanced.py` or `app_mvp.py`
   - Minimalists → `app.py`

2. **Create desktop shortcut:**
   - Right-click `run_enhanced.bat`
   - Create shortcut
   - Move to desktop
   - Rename to "Mandala Laser App"

3. **Prepare test images:**
   - High-contrast photo for engraving
   - Simple logo for vector cutting
   - Circular mandala for multi-layer

4. **Test with real laser cutter:**
   - Start with simple vector cut
   - Then try photo engraving on scrap
   - Finally attempt multi-layer project

---

## Support & Resources

### Documentation:
- [README.md](README.md) - Main overview
- [QUICK_START.md](QUICK_START.md) - Fast-track guide
- [VERSION_COMPARISON.md](VERSION_COMPARISON.md) - Feature comparison
- [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) - Research & features

### External Resources:
- Streamlit Docs: https://docs.streamlit.io
- LightBurn Forum: https://forum.lightburnsoftware.com
- OpenCV Tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

### Troubleshooting:
- Check all dependencies are installed
- Verify Python version (3.9+)
- Clear browser cache if UI looks broken
- Restart Streamlit if app becomes unresponsive

---

## 🎉 Ready to Test!

**Quick Test Command:**
```bash
python -m streamlit run app_enhanced.py
```

**Or simply double-click:**
`run_enhanced.bat`

**Then open your browser to:**
`http://localhost:8501`

---

**Happy Testing!** 🔷✨

If everything works correctly, you're ready to create amazing laser-cut art!
