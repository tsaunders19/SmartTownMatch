# SmartTownMatch

SmartTownMatch CS539 ML Project

> ⚠️ **Startup Warning**  
> The React frontend may load before the Flask backend is fully initialized. **Wait until the backend terminal prints `Running on http://127.0.0.1:5000/` (or similar) before interacting with the site.**

## TODO IMPORTANT (Frontend)

- [ ] Implement a loading / splash screen that checks backend readiness before showing the main UI.

## Overview

Project structure:

- The **backend** is a Python Flask server that handles data processing, clustering, and scoring.
- The **frontend** is a React application that provides an interface to interact with the recommendation engine.

## Prerequisites

Before you begin, you need to have the following installed on your system. Your specific steps might vary slightly depending on your operating system.

1. **Python**: Required for the backend server.
    - Download Python from [python.org](https://www.python.org/downloads/).
    - During installation, make sure to check the box that says **"Add Python to PATH"**.

2. **Node.js**: Required for the frontend application.
    - Download Node.js from [nodejs.org](https://nodejs.org/en/download/).
    - This will also install `npm` (Node Package Manager), which is used to manage frontend dependencies.

## Setup

Follow these steps to get your development environment set up.

### 1. Clone the Repository

First, get a copy of the project on your local machine. If you have `git` installed, you can use the following command:

```bash
git clone <repository-url>
cd SmartTownMatch
```

If you don't have `git`, you can download the project as a ZIP file and extract it.

### 2. Configure API Keys

The application needs a few third-party API keys. **How you set them depends on your platform:**

• **Windows (run_demo.bat)** – You can skip this step. The `run_demo.bat` script will **automatically generate a `.env` file** the first time you run it. It adds the required key names with empty values. If the values remain empty the script should't launch the app.

• **macOS / Linux (Makefile)** – Create the `.env` file yourself **before** running `make backend` / `make frontend`:

```bash
# In the project root
cat > .env <<EOF
CENSUS_API_KEY=
WALK_SCORE_API_KEY=
GOOGLE_PLACES_API_KEY=
EOF
```

You can leave the values blank to get started but you will get limited functionality but the servers will still run. Generally not recommended unless you are testing something specifically

**Where to get the keys (optional but recommended):**

- `CENSUS_API_KEY`: [U.S. Census Bureau signup](https://api.census.gov/data/key_signup.html)
- `WALK_SCORE_API_KEY`: [Walk Score request form](https://www.walkscore.com/professional/api.php)
- `GOOGLE_PLACES_API_KEY`: Enable the *Places API* in Google Cloud and create a key (see [docs](https://developers.google.com/maps/documentation/places/web-service/get-api-key)).

## Running the Application (Easy Way)

Once you have completed the prerequisites and set up your API keys, you can run the application using one of the provided scripts.

### For Windows Users

The `run_demo.bat` script automates the entire setup and launch process **including `.env` creation**.

1. **Double-click** `run_demo.bat` in the project root.
2. On first run the script will:
    - Create or update a `.env` file with placeholder keys (if they were missing).
    - Create a Python virtual environment and install backend requirements.
    - Install frontend npm packages.
    - **If all keys are present**, launch the Flask backend and React frontend in separate windows.
3. **If any API key is blank, the script will stop and prompt you to fill it in**.

Two terminal windows will open: one for the Flask backend and one for the React frontend, but only after valid keys are detected.

### For macOS and Linux Users

The `Makefile` provides simple commands to set up and run the project. You will need two separate terminal windows.

1. **Terminal 1: Start the Backend**
    - Open a terminal and navigate to the project's root directory.
    - Run the `make backend` command. This will set up the virtual environment (if needed) and start the Flask server.

    ```bash
    make backend
    ```

2. **Terminal 2: Start the Frontend**
    - Open a second terminal.
    - Navigate to the project's root directory.
    - Run the `make frontend` command to install dependencies (if needed) and start the React development server.

    ```bash
    make frontend
    ```

The application will open automatically in your browser.

---

## Manual Setup and Execution

If you prefer to set up and run the project manually, or if the scripts above don't work, follow these steps.

### 1. Set Up the Backend

The backend is a Python project. It's best practice to use a virtual environment to manage its dependencies.

1. Open your terminal or command prompt.
2. Navigate to the `backend` directory:

    ```bash
    cd backend
    ```

3. Create a Python virtual environment. This creates an isolated space for the project's Python packages.

    ```bash
    python -m venv .venv
    ```

4. Activate the virtual environment.
    - On **Windows**:

      ```bash
      .venv\Scripts\activate
      ```

    - On **macOS and Linux**:

      ```bash
      source .venv/bin/activate
      ```

5. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### 2. Set Up the Frontend

The frontend is a React project.

1. Open a **new** terminal or command prompt (leave the backend terminal running).
2. Navigate to the `frontend` directory:

    ```bash
    cd frontend
    ```

3. Install the required Node.js packages:

    ```bash
    npm install
    ```

### 3. Running the Application Manually

To run the application, you need to start both the backend and frontend servers in separate terminals.

#### Start the Backend Server

1. Make sure you are in the `backend` directory and your virtual environment is activated (you should see `(.venv)` in your terminal prompt).
2. Run the following command:

    ```bash
    flask run
    ```

3. The backend server will start, typically on `http://127.0.0.1:5000`. Keep this terminal window open.

#### Start the Frontend Application

1. Open the **second** terminal you used for the frontend setup.
2. Make sure you are in the `frontend` directory.
3. Run the following command:

    ```bash
    npm start
    ```

4. This will automatically open the application in your default web browser, usually at `http://localhost:3000`.

You should now have the SmartTownMatch application running locally! The React frontend will automatically connect to the Flask backend.
