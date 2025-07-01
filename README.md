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

## Usage
To get drink suggestions:
1. Navigate to the application in your web browser.
2. Input your available ingredients in the provided field.
3. Submit the form to receive a list of possible drink combinations.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.