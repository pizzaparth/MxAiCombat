import express from "express";
import multer from "multer";
const upload = multer({ dest: "uploads/" });
import { aiDecision } from "../services/decision.service.js";

const router = express.Router();

router.post("/decision",upload.single("file"), aiDecision);

export default router;
