from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.main
import app.routes.kb
import app.clients.pinecone_client

print("IMPORT_OK")
