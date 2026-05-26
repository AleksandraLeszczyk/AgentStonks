from dotenv import load_dotenv

load_dotenv()

from marketview.ui import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=("user", "pass"),
    )
