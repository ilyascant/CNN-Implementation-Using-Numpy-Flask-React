import React, { useState, useRef, useEffect } from "react";
import CanvasPixel from "./CanvasPixel";
import "./Canvas.css";

const SQRT2 = 1.41421356237;

const Canvas = ({ onGuess, setGuessRates, setHighestGuessIndex }) => {
  const size = 28;
  const [pixels, setPixels] = useState(Array(size * size).fill(0));
  const [changeCount, setChangeCount] = useState(0);
  const [brushSize, setBrushSize] = useState(1); // Initial brush size
  const debounceTimeout = useRef(null);

  useEffect(() => {
    return () => {
      if (debounceTimeout.current) {
        clearTimeout(debounceTimeout.current);
      }
    };
  }, []);

  const handleAction = async () => {
    const pixelElements = document.querySelectorAll(".pixel.drawing");
    pixelElements.forEach((pixelElement) => {
      pixelElement.classList.remove("drawing");
    });

    const guess = await onGuess(pixels);
  };

  const clearCanvas = () => {
    if (debounceTimeout.current) {
      clearTimeout(debounceTimeout.current);
      setChangeCount(0);
    }

    setPixels(Array(size * size).fill(0));
    setHighestGuessIndex(null);
    setGuessRates(Array(10).fill(0.0));
  };

  const predict = async () => {
    if (debounceTimeout.current) {
      clearTimeout(debounceTimeout.current);
      setChangeCount(0);
    }

    const guess = await onGuess(pixels);
  };

  const togglePixel = (index) => (e) => {
    const newPixels = [...pixels];
    const row = Math.floor(index / size);
    const col = index % size;

    for (let i = Math.max(0, row - brushSize); i <= Math.min(size - 1, row + brushSize); i++) {
      for (let j = Math.max(0, col - brushSize); j <= Math.min(size - 1, col + brushSize); j++) {
        const maxDistance = brushSize * SQRT2;
        const distance = Math.sqrt((i - row) ** 2 + (j - col) ** 2);
        const intensity = (maxDistance - distance) / brushSize;
        const pixelIndex = i * size + j;

        if (intensity > 0.25) {
          newPixels[pixelIndex] = Math.max(0, Math.min(255, newPixels[pixelIndex] + intensity * 255));
        }
      }
    }

    setPixels(newPixels);
    setChangeCount((prevCount) => {
      const newCount = prevCount + 1;
      if (newCount % 10 === 0) {
        if (debounceTimeout.current) {
          clearTimeout(debounceTimeout.current);
        }
        debounceTimeout.current = setTimeout(handleAction, 1000);
      }
      return newCount;
    });
  };

  return (
    <div>
      <div className="canvas">
        {pixels.map((pixelIntensity, index) => (
          <CanvasPixel
            key={index}
            index={index}
            intensity={pixelIntensity}
            onMouseDown={(e) => togglePixel(index)(e)}
            onMouseOver={(e) => e.buttons === 1 && togglePixel(index)(e)}
          />
        ))}
      </div>
      <button className="predict-button" onClick={predict}>
        Predict
      </button>
      <button className="clear-button" onClick={clearCanvas}>
        Clear Canvas
      </button>
    </div>
  );
};

export default Canvas;
