import React from "react";

function Input({ type = "text", value, onChange, placeholder = "", className = "", name, id }) {
  return (
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className={className}
      name={name}
      id={id}
    />
  );
}

export default Input;
