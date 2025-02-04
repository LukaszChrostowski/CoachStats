import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { styles } from "../style";

import { logo} from "../assets";

const Navbar = () => {
  const [active, setActive] = useState("");
  const [toggle, setToggle] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      if (scrollTop > 100) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);

    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={`${
        styles.paddingX
      } w-full flex items-center py-5 fixed top-0 z-20 left-0 ${
        scrolled ? "bg-green" : "bg-green"
      }`}
    >
      <div className='w-full flex items-center max-w-7xl mx-auto'>
        <Link
          to='/'
          className='flex items-center gap-2'
          onClick={() => {
            setActive("");
            window.scrollTo(0, 0);
          }}
        >
          <img src={logo} alt='logo' width="100" height="100" className="logo img-fluid"/>
          <span className="brand-text" style={{ fontSize: '24px', color: '#99FFCC' }}> 
                    CoachStats
                </span> 
        </Link>

      </div>
    </nav>
  );
};

export default Navbar;
