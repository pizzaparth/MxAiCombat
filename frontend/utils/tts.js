import say from "say";
import fs from "fs";
import path from "path";

export const generateTTS = (text) => {
  const filePath = path.join("public", "tts.wav");

  return new Promise((resolve, reject) => {
    say.export(text, null, 1.0, filePath, (err) => {
      if (err) reject(err);
      else resolve(filePath);
    });
  });
};
