import React, { useState } from "react";
import "./App.css";
import Canvas from "./Canvas";
import axios from "axios";

function App() {
  const [guessRates, setGuessRates] = useState(Array(10).fill(0.0));
  const [highestGuessIndex, setHighestGuessIndex] = useState(null);

  const handleGuess = async (pixelData) => {
    try {
      const guess = await axios.post("http://localhost:5000/predict", {
        pixels: pixelData,
      });

      const data = guess?.data;
      if (data) {
        const probabilities = data.all_probs;
        const guessedDigit = data.prediction;

        const newGuessRates = probabilities.map((probability) => Math.min(probability * 100, 100));

        setGuessRates(newGuessRates);
        setHighestGuessIndex(guessedDigit);

        return data;
      }
    } catch (error) {
      console.error("Prediction request failed:", error);
    }

    return null;
  };

  return (
    <div className="App">
      <div className="guess-rate-container">
        {guessRates.map((rate, index) => (
          <div key={index} className="digit-container">
            <div className={`digit ${index === highestGuessIndex ? "highlight" : ""}`}>{index}</div>
            <div className="progress-bar">
              <div
                className={`progress ${index === highestGuessIndex ? "highlight" : ""}`}
                style={{ height: `${rate}%` }}
              />
            </div>
            <div className={`percentage ${index === highestGuessIndex ? "highlight" : ""}`}>{rate.toFixed(2)}%</div>
          </div>
        ))}
      </div>
      <Canvas onGuess={handleGuess} setGuessRates={setGuessRates} setHighestGuessIndex={setHighestGuessIndex} />
    </div>
  );
}

export default App;
