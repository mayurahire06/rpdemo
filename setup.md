# 🔧 FRAUD DETECTION SYSTEM - SETUP & DEBUG GUIDE

## ❌ YOUR CURRENT ISSUE:
```
HTTP 422 - Unprocessable Entity
Fraud Probability: undefined
```
**ROOT CAUSE**: The frontend is sending category/state values that the backend encoders don't recognize.

---

## ✅ STEP 1: CHECK YOUR ACTUAL DATA

Run this command to see what values are in your CSV:
```bash
python debug_dataset.py
```

This will print:
- ✓ Exact category names
- ✓ Exact state names  
- ✓ Sample fraud vs legitimate transactions
- ✓ Amount statistics

**Copy the output and share it here!**

---

## ✅ STEP 2: USE FIXED FILES

Replace your files with these:

### Backend:
```bash
cp app_fixed.py app.py
```

### Frontend:
```bash
cp index_fixed.html index.html
```

### Trainer (optional - if you need to retrain):
```bash
cp trainer_fixed.py trainer.py
python trainer_fixed.py
```

---

## ✅ STEP 3: RUN THE SYSTEM

### Terminal 1 - Start Backend:
```bash
python app.py
```
You should see:
```
✅ Models loaded successfully!
✅ Encoders loaded successfully!
📋 Valid Categories: ['category1', 'category2', ...]
📋 Valid Genders: ['M', 'F', ...]
📋 Valid States: ['TX', 'CA', 'NY', ...]
```

### Terminal 2 - Open Frontend:
```bash
# On Windows
start index.html

# On Mac
open index.html

# On Linux
xdg-open index.html
```

---

## 📊 WHAT TO ENTER - EXAMPLES

### ✅ LEGITIMATE TRANSACTIONS:
```
Amount: 50 - 500
Category: grocery_pos, shopping_pos, food_dining
Gender: M or F (whatever your dataset has)
State: TX, CA, NY, FL (match your dataset exactly)
```

### 🚨 FRAUDULENT TRANSACTIONS:
```
Amount: 2000 - 10000 (very high)
Category: misc_net or unusual categories
Gender: opposite of typical pattern
State: random or unusual state
```

---

## 🔍 DEBUGGING STEPS

### 1️⃣ Check Backend Health
Open browser and go to:
```
http://127.0.0.1:8000/health
```
You should see JSON with valid values.

### 2️⃣ Check Console Errors
- Open **Chrome DevTools** (F12 or Cmd+Option+I)
- Go to **Console** tab
- Look for red error messages
- Look for the payload being sent

### 3️⃣ Check Backend Terminal
Look for:
```
✅ Models loaded successfully!
📨 POST /predict HTTP/1.1" 200 OK  <- Good
📨 POST /predict HTTP/1.1" 422      <- Bad (data mismatch)
```

### 4️⃣ Test with cURL (if API issues)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amt": 150.00,
    "category": "EXACT_CATEGORY_FROM_CSV",
    "gender": "M",
    "state": "TX"
  }'
```

---

## 🚀 IMPROVEMENTS IN FIXED VERSION

### app_fixed.py:
✅ Loads valid values from encoders  
✅ Validates input against those values  
✅ Clear error messages if validation fails  
✅ health endpoint to check valid options  
✅ Better error handling  

### index_fixed.html:
✅ Dynamically loads categories from backend  
✅ Shows backend connection status  
✅ Better error messages  
✅ Input validation  
✅ Shows transaction details in result  
✅ Prettier UI  

### trainer_fixed.py:
✅ Shows what it's encoding  
✅ Prints encoder mappings  
✅ Shows balancing before/after  
✅ Shows model accuracies  
✅ Better structure  

---

## 📝 COMMON ERRORS & FIXES

### Error: "Invalid category: 'grocery_pos'"
**Solution**: Run `debug_dataset.py` to see exact category names in your CSV

### Error: "Invalid state: 'TX'"
**Solution**: States might be lowercase or different format. Check debug output.

### Error: "Backend not responding"
**Solution**: Make sure app.py is running in another terminal on port 8000

### Error: "Fraud Probability: undefined"
**Solution**: Check browser console (F12) for the actual error message

### Port 8000 already in use
**Solution**: 
```bash
# Kill process using port 8000
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# On Mac/Linux:
lsof -ti:8000 | xargs kill -9
```

---

## 🎯 NEXT STEPS

1. Run `python debug_dataset.py`
2. Share the output
3. Copy fixed files
4. Run `python app.py`
5. Open `index.html`
6. Test with sample transactions
7. Check browser console if issues

---

## ❓ QUESTIONS TO ANSWER

1. What are your **exact category values** from CSV?
2. What are your **exact state values** from CSV?
3. Are genders **'M', 'F'** or something else?
4. What's the **typical amount range**?
   - Legitimate: $X - $Y
   - Fraudulent: $A - $B

Once you share the debug output, I can create the **perfect matching configuration**! 

🎯 Let me know the results!