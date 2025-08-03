# test_google_api.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

print("--- Starting Google API Key Test ---")

# 1. Load the .env file
load_dotenv()

# 2. Get the API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ FATAL ERROR: GOOGLE_API_KEY not found in your .env file.")
    print("Please check the file name and the key name.")
else:
    print("✅ API Key found in .env file.")
    
    try:
        # 3. Configure the Google AI client with your key
        genai.configure(api_key=api_key)
        print("✅ Google AI client configured.")

        # 4. Create an instance of the model we want to test
        # Using the latest, most compatible model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("✅ Model 'gemini-1.5-flash-latest' initialized.")

        # 5. Send a simple prompt to the model
        print("\nSending a test prompt to Google AI...")
        response = model.generate_content("What is the speed of light?")
        
        # 6. Print the result
        print("\n--- ✅ SUCCESS! ---")
        print("API Key is working correctly. Response from Google:")
        print(response.text)

    except Exception as e:
        print("\n--- ❌ TEST FAILED ---")
        print("The API key or project setup is incorrect. Here is the exact error from Google:")
        print("-------------------------------------------------")
        import traceback
        traceback.print_exc()
        print("-------------------------------------------------")

print("\n--- Test Script Finished ---")