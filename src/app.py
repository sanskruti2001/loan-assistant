import secrets
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
from .retrieve_and_answer import answer_question

# Initialize FastAPI app with a custom title
app = FastAPI(title="Loan Product Assistant (BoM)")

# Add session middleware with a random secret key generated on each server start
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(16))

# Define paths for static files and templates
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Mount the static directory for serving CSS, JS, and image files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize Jinja2 template engine for rendering HTML pages
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Route for the home page (GET request)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Retrieve chat messages from the user's session (if any)
    messages = request.session.get("messages", [])
    # Render the index.html template with messages
    return templates.TemplateResponse("index.html", {"request": request, "messages": messages})

# Route to handle user question submission (POST request)
@app.post("/", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):
    # Retrieve previous messages from session
    messages = request.session.get("messages", [])

    try:
        # Call the function to get an answer for the user's question
        result = answer_question(question, top_k=5)
        # Extract the answer text from the result or show a fallback message
        answer_text = result.get("answer", "Sorry, I couldn’t find that.")
    except Exception as e:
        # Handle errors and show error message in the chat
        answer_text = f"Error: {str(e)}"

    # Append user question and bot answer to session messages
    messages.append({"sender": "user", "text": question})
    messages.append({"sender": "bot", "text": answer_text})
    # Update session with the latest conversation history
    request.session["messages"] = messages

    # Redirect back to the home page to display updated chat
    return RedirectResponse(url="/", status_code=303)

# Route to clear the chat session
@app.get("/clear")
async def clear_session(request: Request):
    # Remove all stored session data
    request.session.clear()
    # Redirect back to the home page
    return RedirectResponse(url="/", status_code=303)
