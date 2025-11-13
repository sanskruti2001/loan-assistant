# from pathlib import Path
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from starlette.middleware.sessions import SessionMiddleware
# from .retrieve_and_answer import answer_question

# app = FastAPI(title="Loan Product Assistant (BoM)")

# # Add session middleware (for chat history)
# app.add_middleware(SessionMiddleware, secret_key="supersecretkey")

# # Paths
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# STATIC_DIR = PROJECT_ROOT / "static"
# TEMPLATES_DIR = PROJECT_ROOT / "templates"

# app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     # Get chat history from session (if available)
#     messages = request.session.get("messages", [])
#     return templates.TemplateResponse("index.html", {"request": request, "messages": messages})


# @app.post("/", response_class=HTMLResponse)
# async def ask(request: Request, question: str = Form(...)):
#     messages = request.session.get("messages", [])

#     try:
#         result = answer_question(question, top_k=5)
#         answer_text = result.get("answer", "Sorry, I couldn’t find that.")
#     except Exception as e:
#         answer_text = f"Error: {str(e)}"

#     # Append user and bot messages
#     messages.append({"sender": "user", "text": question})
#     messages.append({"sender": "bot", "text": answer_text})

#     # Save to session
#     request.session["messages"] = messages

#     # Redirect to avoid re-submission
#     return RedirectResponse(url="/", status_code=303)





import secrets
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
from .retrieve_and_answer import answer_question

app = FastAPI(title="Loan Product Assistant (BoM)")

# Random secret key per server start
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(16))

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    messages = request.session.get("messages", [])
    return templates.TemplateResponse("index.html", {"request": request, "messages": messages})

@app.post("/", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):
    messages = request.session.get("messages", [])

    try:
        result = answer_question(question, top_k=5)
        answer_text = result.get("answer", "Sorry, I couldn’t find that.")
    except Exception as e:
        answer_text = f"Error: {str(e)}"

    messages.append({"sender": "user", "text": question})
    messages.append({"sender": "bot", "text": answer_text})
    request.session["messages"] = messages

    return RedirectResponse(url="/", status_code=303)

@app.get("/clear")
async def clear_session(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)
