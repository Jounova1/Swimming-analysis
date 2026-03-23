import React from "react";

function Button({ type = "button", onClick, children, className = "", disabled = false }) {
  return (
    <button
      type={type}
      onClick={onClick}
      className={className}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

export default Button;
