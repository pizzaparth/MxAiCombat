import FormData from "form-data";
import fs from "fs";
import axios from "axios";

async function whisperSTT(audioPath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(audioPath));

  const res = await axios.post("http://127.0.0.1:5000/stt", form, {
    headers: form.getHeaders()
  });

  return res.data.text;
}
