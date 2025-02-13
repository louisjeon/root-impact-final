const Company = require("../models/company.model");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcrypt");

// 회원가입 (User 생성)
exports.signup = async (req, res) => {
  req.body.token = "";
  const { id, password } = req.body;

  const company = await Company.findOne({ id });

  if (company) {
    return res.status(400).json({ success: false, error: "ID already exists" });
  }

  const hashedPassword = await bcrypt.hash(password, 10);

  req.body.password = hashedPassword;

  const newCompany = new Company(req.body);

  try {
    await newCompany.save();
    res.status(200).json({ success: true, newCompany });
  } catch (err) {
    console.error(err);
    res
      .status(400)
      .json({ success: false, error: "Signup failed", details: err });
  }
};

// 모든 유저 조회
exports.getAllCompanies = async (req, res) => {
  try {
    const companies = await Company.find();
    res.status(200).json(companies);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch companies", details: err });
  }
};

// useId로 유저 조회
exports.getCompanyById = async (req, res) => {
  const { companyId } = req.params;

  try {
    const company = await Company.findById(companyId);
    if (!company) {
      return res.status(404).json({ error: "Company not found" });
    }
    res.status(200).json(company);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch company", details: err });
  }
};

exports.getCompanyByName = async (req, res) => {
  const { companyName } = req.params;

  try {
    const company = await Company.find({
      name: companyName,
    });
    if (!company) {
      return res.status(404).json({ error: "Company not found" });
    }
    res.status(200).json(company);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch company", details: err });
  }
};

exports.updateCompany = async (req, res) => {
  try {
    const { name, id, email, dateOfFoundation, introduction, jobPosts } =
      req.body;
    const company = await Company.findByIdAndUpdate(
      req.params.companyId,
      { name, id, email, dateOfFoundation, introduction, jobPosts },
      { new: true }
    );
    if (!company) {
      console.log(`Company not found with ID: ${req.params.companyId}`);
      return res.status(404).json({ error: "Company not found" });
    }
    console.log(`Updated company: ${company._id}`);
    res.status(200).json(company);
  } catch (error) {
    console.error("Error updating company:", error.message);
    res.status(400).json({ error: "Failed to update company" });
  }
};

// 로그인
exports.login = async (req, res, next) => {
  const { id, password } = req.body;
  const company = await Company.findOne({ id });

  if (!company) {
    return res.status(401).json({ message: "ID doesn't exist." });
  }

  const isPasswordValid = await bcrypt.compare(password, company.password);

  if (!isPasswordValid) {
    return res.status(401).json({ message: "Invalid password." });
  }

  const token = jwt.sign({ id: company._id }, process.env.JWT_KEY);
  res.json({ token, company });
};

// 유저 삭제
exports.deleteCompany = async (req, res) => {
  try {
    const company = await Company.findByIdAndDelete(req.params.companyId);
    if (!company) {
      console.log(`Company not found with ID: ${req.params.companyId}`);
      return res.status(404).json({ error: "Company not found" });
    }
    console.log(`Deleted company with ID: ${company._id}`); // 삭제된 댓글 ID 출력
    res.status(200).json({ message: "Company deleted successfully" });
  } catch (error) {
    console.error("Error deleting company:", error.message); // 에러 메시지 출력
    res.status(500).json({ error: "Failed to delete company" });
  }
};
