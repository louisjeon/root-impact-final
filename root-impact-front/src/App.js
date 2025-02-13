import React from "react";
import Header from "./components/Header";
import Home from "./menu/Home";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import About from "./menu/About";
import Products from "./menu/Products";
import News from "./menu/News";
import Board from "./menu/Board";
import Signup from "./menu/Signup";
import Login from "./menu/Login";
import Post from "./board/Post";
import WritePost from "./board/WritePost";
import { AuthProvider } from "./AuthContext";
import { ThemeProvider } from "styled-components";
import theme from "./theme";
import MyPage from "./menu/MyPage";
import CompanyMyPage from "./menu/CompanyMyPage";
import Resumes from "./menu/Resumes";
import Resume from "./menu/Resume";
import AppliedJobs from "./menu/AppliedJobs";
import AppliedJob from "./menu/AppliedJob";
import Companies from "./menu/Companies";
import Company from "./menu/Company";
import Candidates from "./menu/Candidates";
import Candidate from "./menu/Candidate";
import Consulting from "./menu/Consulting";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "about",
    element: <About />,
  },
  {
    path: "products",
    element: <Products />,
  },
  {
    path: "news",
    element: <News />,
  },
  {
    path: "board",
    element: <Board />,
  },
  {
    path: "resumes",
    element: <Resumes />,
  },
  {
    path: "resume",
    element: <Resume />,
  },
  {
    path: "login",
    element: <Login />,
  },
  {
    path: "signup",
    element: <Signup />,
  },
  {
    path: "post",
    element: <Post />,
  },
  {
    path: "writePost",
    element: <WritePost />,
  },
  {
    path: "myPage",
    element: <MyPage />,
  },
  {
    path: "companyMyPage",
    element: <CompanyMyPage />,
  },
  {
    path: "appliedJobs",
    element: <AppliedJobs />,
  },
  {
    path: "appliedJob",
    element: <AppliedJob />,
  },
  {
    path: "companies",
    element: <Companies />,
  },
  {
    path: "company",
    element: <Company />,
  },
  {
    path: "candidates",
    element: <Candidates />,
  },
  {
    path: "candidate",
    element: <Candidate />,
  },
  {
    path: "consulting",
    element: <Consulting />,
  },
]);

const App = () => {
  return (
    <AuthProvider>
      <ThemeProvider theme={theme}>
        <Header />
        <RouterProvider router={router} />
      </ThemeProvider>
    </AuthProvider>
  );
};

export default App;
