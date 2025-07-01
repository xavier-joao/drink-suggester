# Drink Suggester

## Overview
The Drink Suggester is a web application that allows users to input a list of ingredients and receive possible drink combinations based on a comprehensive drinks database. This project utilizes a dataset of drinks to suggest creative and delicious drink options.

## Project Structure
```
drink-suggester
├── src
│   ├── app.py          # Main entry point of the application
│   ├── database
│   │   └── drinks.csv  # Dataset of drinks and their ingredients
│   └── utils
│       └── suggest.py  # Function to suggest drinks based on ingredients
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/drink-suggester.git
   cd drink-suggester
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/app.py
   ```

4. Access the web application in your browser at `http://localhost:5000`.

## Running with Docker

1. Build the Docker image:
   ```
   docker build -t drink-suggester .
   ```

2. Run the container:
   ```
   docker run -p 5000:5000 drink-suggester
   ```

3. Access the web application at [http://localhost:5000](http://localhost:5000).

## Usage
To get drink suggestions:
1. Navigate to the application in your web browser.
2. Input your available ingredients in the provided field.
3. Submit the form to receive a list of possible drink combinations.

## API Usage

You can also use the `/suggest` API endpoint directly:

**Example cURL:**
```sh
curl -X POST http://localhost:5000/suggest \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["vodka", "lime juice", "triple sec"]}'
```

**Response:**
```json
{
  "probability": 0.87,
  "similar_drinks": [
    {
      "name": "cosmopolitan",
      "ingredients": ["vodka", "lime juice", "triple sec", "cranberry juice"],
      "similarity": 0.95
    }
    // ...
  ]
}
```