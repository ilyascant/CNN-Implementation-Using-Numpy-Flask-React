import React from "react";

const CanvasPixel = ({ intensity, index, onMouseDown, onMouseOver }) => {
  const row = Math.floor(index / 28);
  const col = index % 28;

  return (
    <div
      id={`${row},${col}`}
      className="pixel"
      style={{
        backgroundColor: `rgb(${intensity}, ${intensity}, ${intensity})`,
        "--intensity": intensity,
      }}
      onMouseDown={onMouseDown}
      onMouseOver={onMouseOver}
    />
  );
};

export default CanvasPixel;
