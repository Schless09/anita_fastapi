import pickle
import base64
import io
import os

# Ensure this path is correct relative to where you run the script
token_path = 'token.pkl'
output_file = 'token_b64.txt' # Optional: output file name

if not os.path.exists(token_path):
    print(f"Error: {token_path} not found. Please generate it first using auth.py.")
else:
    try:
        with open(token_path, 'rb') as token_file:
            # Read the binary content
            token_bytes = token_file.read()
            # Base64 encode it
            encoded_bytes = base64.b64encode(token_bytes)
            # Decode to a string for saving/printing
            encoded_string = encoded_bytes.decode('utf-8')

            print("\n-----BEGIN NEW BASE64 TOKEN-----")
            print(encoded_string)
            print("-----END NEW BASE64 TOKEN-----\n")

            # Write directly to the file, replacing its content
            with open(output_file, 'w') as outfile:
               outfile.write(encoded_string)
            print(f"Successfully updated {output_file} with the new token.")

    except Exception as e:
        print(f"An error occurred: {e}")
