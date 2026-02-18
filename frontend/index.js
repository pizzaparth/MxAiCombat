import express from "express";
import cors from "cors";
import aiRoutes from "./routes/ai.routes.js";

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.static("public"));


app.use("/ai", aiRoutes);

app.get("/", (req, res) => {
  res.send("server at localhost: http://localhost:3000");
});

app.listen(3000, () => {
  console.log("Server started on port 3000");
});
