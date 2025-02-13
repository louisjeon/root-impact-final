import React, { useEffect, useState } from "react";
import styled from "styled-components";
import { useAuth } from "../AuthContext";

const MenuWrapper = styled.div`
  display: flex;
  flex: 1;
  justify-content: flex-end;
  aling-items: center; /* 요소들을 세로 중앙 정렬 */
;
  a {
    text-decoration: none;
  }

  a:link,
  a:visited {
    color: inherit;
  }

  a:hover {
    color: gray;
  }

  h5:hover {
    cursor: pointer;
  }

  .menu {
    display: flex;
    align-items: center;
    justify-content: space-around;
    flex-grow: 1;
    flex: 1;
    margin: 15px 10px 0px 10px;
    max-width: 800px;
    font-size: 20px;
    display: flex;
    flex-direction: row;
    ${"" /* color: #f54458; */}
    justify-content: space-around;

    .username {
      position: relative;
    }

    .infoDropdown {
      position: absolute;
      top: 40px;
      left: -30px;
      z-index: 1000;
      background-color: white;
      padding: 0 20px;
      width: 70px;
      border: 1px solid black;
      border-radius: 10px;

      a {
        height: 50px;
        font-size: 20px;
      }
    }
  }

  .menuDropdown {
    width: 50px;
    height: 50px;
    background-color: white;
    font-size: 50px;
    margin-right: 25px;
    padding: 0;
    color: white;
    position: relative;
    border-radius: 10px;
    background-color: ${({ theme }) => theme.colors.MAIN};

    p {
      margin-top: -10px;
    }

    &:hover {
      cursor: pointer;
    }
  }

  .dropdown {
    position: absolute;
    top: 50px;
    color: black;
    font-size: 25px;
    right: 0;
    background-color: white;
    border: 1px solid black;
    border-radius: 10px;
    width: 160px;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  @media screen and (min-width: 900px) {
    .menuDropdown {
      display: none;
    }
  }

  @media screen and (max-width: 899px) {
    .menu {
      display: none;
    }
  }
`;

// 새로운 버튼 스타일
const StyledButton = styled.a`
  display: flex;
  align-items: center;
  justigy-content: center;
  background-color: ${({ theme }) => theme.colors.MAIN};
  color: white !important;
  font-size: 18px;
  font-weight: bold;
  text-align: center;
  height: 50px;
  padding: 0px 20px;
  border-radius: 25px;
  text-decoration: none;
  transition: background-color 0.3s ease;
`;


const Menu = () => {
  const { user, logout, isCompanyUser, setCompanyUser } = useAuth();
  const [showLogout, setShowLogout] = useState(false);
  const [showDropdown, setShowDropdown] = useState(true);

  useEffect(() => {
    setShowDropdown(false);
  }, [window.location.protocol + "//" + window.location.host]);

  return (
    <MenuWrapper>
      <div className="menu">
        <a href="/board">
          <h5>{isCompanyUser() && "전체 "}채용 공고</h5>
        </a>

        {isCompanyUser() ? (
          <a href="/board?mine=true">
            <h5>나의 채용 공고</h5>
          </a>
        ) : (
          <>
            <a href="/resumes">
              <h5>이력서 관리</h5>
            </a>
            <a href="/appliedJobs">
              <h5>지원 현황</h5>
            </a>
          </>
        )}

        <a href="/news">
          <h5>기업 뉴스</h5>
        </a>
        {isCompanyUser() ? (
          <>
            <a href="/candidates">
              <h5>인재 풀 탐색</h5>
            </a>
            <a href="/consulting">
              <h5>컨설팅 요청</h5>
            </a>
          </>
        ) : (
          <a href="/companies">
            <h5>회사 탐색</h5>
          </a>
        )}
        {user() ? (
          <>
            <h5 className="username" onClick={() => setShowLogout(!showLogout)}>
              {user().name}님
              {showLogout && (
                <div className="infoDropdown">
                  {isCompanyUser() ? (
                    <a href="/companyMyPage">
                      <h5 id="mypage">Company Page</h5>
                    </a>
                  ) : (
                    <a href="/mypage">
                      <h5 id="mypage">My Page</h5>
                    </a>
                  )}
                  <a href="/login">
                    <h5 id="logout" onClick={logout}>
                      Logout
                    </h5>
                  </a>
                </div>
              )}
            </h5>
          </>
        ) : (
          <>
            {isCompanyUser() ? (
              <h5
                onClick={() => {
                  setCompanyUser(false);
                  window.location.pathname = "/";
                }}
              >
                오작교(구직)
              </h5>
            ) : (
              <h5
                onClick={() => {
                  setCompanyUser(true);
                  window.location.pathname = "/";
                }}
              >
                오작교(기업)
              </h5>
            )}

            <StyledButton href="/login">JOIN US</StyledButton>
          </>
        )}
      </div>
      <button
        className="menuDropdown"
        onClick={() => setShowDropdown(!showDropdown)}
      >
        <p>&#9776;</p>
        {showDropdown && (
          <div className="dropdown">
            <a href="/board">
              <h5>{isCompanyUser() && "전체 "}채용 공고</h5>
            </a>

            {user() && isCompanyUser() ? (
              <a href="/board?mine=true">
                <h5>나의 채용 공고</h5>
              </a>
            ) : (
              <>
                <a href="/resumes">
                  <h5>이력서 관리</h5>
                </a>
                <a href="/appliedJobs">
                  <h5>지원 현황</h5>
                </a>
              </>
            )}

            <a href="/news">
              <h5>기업 뉴스</h5>
            </a>
            {user() && isCompanyUser() ? (
              <>
                <a href="/candidates">
                  <h5>인재 풀 탐색</h5>
                </a>
                <a href="/consulting">
                  <h5>컨설팅 요청</h5>
                </a>
              </>
            ) : (
              <a href="/companies">
                <h5>회사 탐색</h5>
              </a>
            )}

            {user() ? (
              <>
                {isCompanyUser() ? (
                  <a href="/companyMyPage">
                    <h5 id="mypage">Company Page</h5>
                  </a>
                ) : (
                  <a href="/mypage">
                    <h5 id="mypage">My Page</h5>
                  </a>
                )}
                <a href="/login">
                  <h5 id="logout" onClick={logout}>
                    Logout
                  </h5>
                </a>
              </>
            ) : (
              <>
                {isCompanyUser() ? (
                  <h5
                    onClick={() => {
                      setCompanyUser(false);
                      window.location.pathname = "/";
                    }}
                  >
                    오작교(구직)
                  </h5>
                ) : (
                  <h5
                    onClick={() => {
                      setCompanyUser(true);
                      window.location.pathname = "/";
                    }}
                  >
                    오작교(기업)
                  </h5>
                )}
                <a href="/login">
                  <h5>로그인/회원가입</h5>
                </a>
              </>
            )}
          </div>
        )}
      </button>
    </MenuWrapper>
  );
};

export default Menu;
