import axios from "axios";
import say from "say";
import fs from "fs";
import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import FormData from "form-data";

const execFileAsync = promisify(execFile);

// 🔊 Whisper STT helper
async function whisperSTT(audioPath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(audioPath));

  const res = await axios.post("http://127.0.0.1:5000/stt", form, {
    headers: form.getHeaders()
  });

  return res.data.text;
}

export const aiDecision = async (req, res) => {
  const { temperature, spo2, heart_rate, voice_text } = req.body;

  let finalText = voice_text || "";

  // 🎙️ If audio file sent → Whisper
  if (req.file) {
    try {
      finalText = await whisperSTT(req.file.path);
      fs.unlinkSync(req.file.path); // cleanup
      console.log("Whisper text:", finalText);
    } catch (err) {
      console.log("Whisper failed:", err.message);
    }
  }

  let response = {
    allow_dispense: true,
    medicine: null,
    motor: 1,
    message: "No medicine suggested",
    audio_url: "http://10.107.170.85:3000/tts.wav",
    txt: finalText
  };






  // 🧠 AI DECISION LOGIC

  
  if (temperature >= 38 && finalText.toLowerCase().includes("fever")) {
    response.allow_dispense = true;
    response.medicine = "Paracetamol";
    response.motor = 6; 
    response.message = "Welcome to  meth x ai. Fever detected, take Paracetamol. Thank you.";



















    

    // 🔊 TTS
    const publicDir = path.resolve(process.cwd(), "public");
    fs.mkdirSync(publicDir, { recursive: true });

    const aiffPath = path.join(publicDir, "tts.aiff");
    const wavPath = path.join(publicDir, "tts.wav");

    try {
      await new Promise((resolve, reject) => {
        say.export(response.message, null, 1.0, aiffPath, err =>
          err ? reject(err) : resolve()
        );
      });

      await execFileAsync("afconvert", [
        "-f", "WAVE",
        "-d", "LEI16@16000",
        aiffPath,
        wavPath
      ]);

      fs.unlinkSync(aiffPath);

      response.audio_url = `http://192.168.1.5:3000/tts.wav`; // 🔴 YOUR PC IP

    } catch (err) {
      console.log("⚠️ TTS failed:", err.message);
    }
  }

  // 🐍 Send log to Python (optional)
  try {
    await axios.post("http://127.0.0.1:5000/test", {
      temperature,
      spo2,
      heart_rate,
      text: finalText,
      decision: response
    });
  } catch {
    console.log("⚠️ Python not reachable");
  }

  // ✅ Send back to ESP32
  res.json(response);
};
