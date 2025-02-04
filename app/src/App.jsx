import { BrowserRouter } from "react-router-dom";
import { Hero, Navbar } from "./components";


const App = () => {
  


  return (
    <BrowserRouter>
      <div className='relative z-0 bg-lightgreen'>
        <div className='bg-hero-pattern bg-cover bg-no-repeat'>
          <Navbar/>
        </div>
        <div className='relative z-0 w-full'>
         <Hero/>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;