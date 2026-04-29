import base64
import urllib.request
import json

graph = """graph TD
    classDef file fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#ecf0f1;
    classDef module fill:#2980b9,stroke:#2980b9,stroke-width:2px,color:#ffffff;
    classDef frontend fill:#8e44ad,stroke:#8e44ad,stroke-width:2px,color:#ffffff;

    D1[bank_dea_dataset.csv]:::file --> DEA[dea_model.py<br>Optimization Engine]:::module
    DEA --> D2[dea_results.csv]:::file

    D1 --> Viz[visualization.py<br>Visualization Engine]:::module
    D2 --> Viz
    
    D1 --> Sim[simulation.py<br>Recommendation Engine]:::module
    D2 --> Sim
    
    D1 --> Intel[intelligence.py<br>Intelligence Platform]:::module
    D2 --> Intel
    Sim --> Intel

    Viz --> App[app.py<br>Streamlit Dashboard]:::frontend
    Sim --> App
    Intel --> App"""

state = {"code": graph, "mermaid": {"theme": "default"}}
json_str = json.dumps(state).encode('utf-8')
encoded = base64.urlsafe_b64encode(json_str).decode('utf-8')
url = f"https://mermaid.ink/img/{encoded}"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

print("Downloading image from Mermaid API...")
try:
    with urllib.request.urlopen(req) as response, open("block_diagram.png", "wb") as out_file:
        out_file.write(response.read())
    print("Successfully saved block_diagram.png")
except Exception as e:
    print("Error:", e)
