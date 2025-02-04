import React, { useEffect, useState } from "react";
import { styles } from "../style";
import { ComputersCanvas } from "./canvas";
import { Link } from "react-router-dom";

const Hero = () => {
  
  const [gameMinute, setGameMinute] = useState('');

  const [firstShot, setfirstShot] = useState(0);
  const [oneOnOne, SetoneOnOne] = useState(0);
  const [afterAirDuel, SetafterAirDuele] = useState(0);
  const [openGoal, SetOpenGoal] = useState(0);
  const [afterDribbling, SetafterDribbling] = useState(0);
  const [redirect, SetRedirect] = useState(0);

  const handleMinuteChange = (e) => { setGameMinute(e.target.value);
    //sentQuestion();
  }

  const handleFirstShotChange = (event) => {
    setfirstShot(event.target.checked);
    //sentQuestion();
};

const handleOneOnOneChange = (event) => {
    SetoneOnOne(event.target.checked);
    //sentQuestion();
  };

const handleAfterAirDuelChange = (event) => {
    SetafterAirDuele(event.target.checked);
    //sentQuestion();
  };

const handleOpenGoalChange = (event) => {
    SetOpenGoal(event.target.checked);
    //sentQuestion();
  };

const handleAfterDribblingChange = (event) => {
    SetafterDribbling(event.target.checked);
    //sentQuestion();
  };

const handleRedirectChange = (event) => {
    SetRedirect(event.target.checked);
    //sentQuestion();
  };

  
  const [data, setData] = useState("");
  //zmienne globalne
  //zmienna mówiąca który przycisk jest aktywny, jest ona wykorzystywana 
  //przez listener nanoszący zawodników na boisko. 1 - strzelec , 2 - bramkarz , 3 - obronca, 4 - napasnik. zmienna przez aktywacje przycskówk bb1, bb2, bb3 i bb4

  let active_bbt = "bbt1";
  const [number_of_defenders, setNumberOfDevenders] = useState(()=>{
    return 0;
  });

  const [number_of_strikers, setNumberOfStrikers] = useState(() => {
    return 0;
  })
  const [number_of_goalkeepers, setNumberOfGoalkeppers] = useState(()=>{
    return 0;
  });
  const [number_of_shooters,setNumberOfShooters] = useState(() => {
  return 0}
  );

  const number_of_shooters_rev = React.useRef(number_of_shooters)
  const number_of_strikers_rev = React.useRef(number_of_strikers)
  const number_of_goalkeepers_rev = React.useRef(number_of_goalkeepers)
  const number_of_defenders_rev = React.useRef(number_of_defenders)

  // zmienne globalne bedace danymi wejsciowymi do modelu

  var shooter = [0,0]
  var goalkeeper = [0,0]
  
  var stricers = new Array;
  var defenders = new Array;

  const [bodyPart,setBodyPart] = useState('Right Foot');
  const [technique,setTechnique] = useState('Normal');
  const [actionType,setActionType] = useState('Open Play');
  const [shooterPossition,setPossition] = useState('Center Forward');

//zamiana procentowych pozycji na kartezyjskie
  function konwerturX(x){
    var fX = parseFloat(x.slice(0,-1))
    return Math.round((fX*105))/100
  }
  function konwetujY(y){
    var fY = parseFloat(y.slice(0,-1))
    return Math.round(fY*68)/100
  }

  function changePlayer(a){

    document.getElementById("bbt1").style.background = "#99FFCC";
    document.getElementById("bbt2").style.background = "#99FFCC";
    document.getElementById("bbt3").style.background = "#99FFCC";
    document.getElementById("bbt4").style.background = "#99FFCC";
    document.getElementById(a).style.background = "#00CC66";
    active_bbt = a
  }

  // Reset Boiska
  function resetField() {
      var footballField = document.getElementById('footballField');
      var footballs = document.querySelectorAll('.football');
      var list = document.getElementById('list')
      var players = document.querySelectorAll('.player')

      footballs.forEach(function (ball) {
          footballField.removeChild(ball);
      });
      players.forEach(function(element) {
        list.removeChild(element);
      })

      setNumberOfDevenders(0);
      setNumberOfStrikers(0)
      setNumberOfGoalkeppers(0)
      setNumberOfShooters(0)
      setGameMinute('')

      number_of_shooters_rev.current = 0;
      number_of_goalkeepers_rev.current = 0;
      number_of_defenders_rev.current = 0;
      number_of_strikers_rev.current = 0;

      document.getElementById("ex").innerHTML = 0;
  }
  //funkcja dodaje zawodnika do listy zawodnikow. zmienne x oraz y to wspolrzedne a position
  // to pozycja zawodnika 0 - strzelec, 1 - bramkarz, 2 - broniacy, 3 napastnik
  // ball - to odnosnik do punktu na boisku

  function deletePlayer(ball,player,possition){
    var list = document.getElementById('list');
    var bojo = document.getElementById('footballField');
    list.removeChild(player)
    bojo.removeChild(ball)
    if(possition == 1){
      setNumberOfGoalkeppers(number_of_goalkeepers_rev.current -= 1);
    }else if(possition == 2){
      setNumberOfDevenders(number_of_defenders_rev.current -= 1) ;
    }else if (possition == 3){
      setNumberOfStrikers(number_of_strikers_rev.current -= 1);
    }else if(possition == 0){ 
      setNumberOfShooters(number_of_shooters_rev.current -= 1);
    }
    //sentQuestion()
  }
  
  // funkcja zsczytuje pozycje zawodnikow przed wyslaniem zapytania do serwera 
  function loadPlayers(){
    var players = document.querySelectorAll('.player')

    // wyczyszczenie list zawierajacych lokalizacje zawodnikow 
    defenders = []
    stricers = []
    shooter = null
    goalkeeper = null
    //zebranie iformacji o kazdym z zawodnikow

    players.forEach(function(player){
      var type = player.getAttribute('type')
      
      if(type == 0){
        shooter = player.getAttribute('possition')
      }else if(type == 1 ){
        goalkeeper = player.getAttribute('possition')
      }else if(type == 2){
        defenders.push(player.getAttribute('possition'))
      }else {
        stricers.push(player.getAttribute('possition'))
      }
    
    })
  }
  function addPlayer(possition, ball){ 
    //Zmiana stylu w zaleznosci czy zawodnik jest obronca/napastnikiem iyp.
    var pName = "Strzelec"
    //kolor kropki
    // kolor tła kafelka
    var pColor = "#fc0303"
    if(possition == 1){
      var pName = "Bramkarz"
      var pColor = "#03e7fc"
      setNumberOfGoalkeppers(number_of_goalkeepers_rev.current += 1);
    }else if(possition == 2){
      var pName = "Obrońca"
      var pColor = "#0324fc"
      setNumberOfDevenders(number_of_defenders_rev.current += 1);
    }else if (possition ==3){
      var pName = "Napastnik"
      var pColor = "#fc6703"
      setNumberOfStrikers(number_of_strikers_rev.current += 1);
    }else{
      setNumberOfShooters(number_of_shooters_rev.current += 1);
    }
    var player = document.createElement('div');
    player.className = 'player';
    player.style.width = "inherit"
    player.setAttribute("type",possition) 
    var list = document.getElementById("list")
    
    // div z nazwa gracza
    var tekst = document.createElement('div')
    tekst.style.fontSize = "12px";
    tekst.innerHTML = pName;
    
    // div z pozycja gracza
    var posytion = document.createElement('div')
    posytion.style.fontSize = "12px"
    posytion.innerHTML = konwerturX(ball.style.left) + " m, " + konwetujY(ball.style.top) + " m";
    player.setAttribute('possition',[konwerturX(ball.style.left),konwetujY(ball.style.top)])
    //div z przyciskiem usuwającym
    var btnDelete = document.createElement('button')
    btnDelete.innerHTML = 'Usuń'
    btnDelete.style.fontSize = "12px"
    btnDelete.style = 'background-color: #FFB266;color:#000000'
    btnDelete.addEventListener("click",function(){deletePlayer(ball,player,possition)},false)
    //dodanie elementów do kafelka
    player.appendChild(tekst)
    player.appendChild(posytion)
    player.appendChild(btnDelete)
    list.appendChild(player)

    player.addEventListener("mouseover",function(){
      ball.style.background = "#42f5d7"
    })
    player.addEventListener("mouseout",function(){
      ball.style.background = pColor;
    })

    ball.addEventListener("mouseover",function podswietl(){
      player.style.background = "#bf8e8e"
      tekst.style.color = "black"
      posytion.style.color = "black"
    })
    ball.addEventListener("mouseout",function (){
      player.style.background = 'rgb(16, 46, 29)';
      tekst.style.color = "white"
      posytion.style.color = "white"
    })

    var bojo = document.getElementById('footballField')

    // listenery pozwalające na przesuwanie punktu
    

    ball.addEventListener("mousedown", function przesuwanie(){
      
      ball.style.background = "orange"
      player.style.background = 'rgb(16, 46, 29)';
      tekst.style.color = "white"
      posytion.style.color = "white"

      bojo.addEventListener("mousemove", function whileMove(ev){
        var bojo = document.getElementById('footballField')
        var bnd = ev.target.getBoundingClientRect()
        // zapis od 0 - 1 w jakim procentowym miejscu boiska użytkownik kliknął
        var x = ((ev.clientX - bnd.left)/bojo.offsetWidth)*100;
        var y = ((ev.clientY - bnd.top)/bojo.offsetHeight)*100;
       
        player.style.background = 'rgb(16, 46, 29)';
        tekst.style.color = "white"
        posytion.style.color = "white"
        
        x = parseFloat(x) -3
        y = parseFloat(y) -3  
        
        if(bojo.parentNode.querySelector(":hover")){
          ball.style.left = x + "%"
          ball.style.top = y + "%"
        }
        //var shooterX = konwerturX(ball.style.left) 
        //var shooterY = konwetujY(ball.style.top)
        posytion.innerHTML =  "x:" + konwerturX(ball.style.left) + "m " + "y:" + konwetujY(ball.style.top) + "m";
        player.setAttribute('possition',[konwerturX(ball.style.left),konwetujY(ball.style.top)]);
      bojo.addEventListener("mouseup", function afterUp(){
          ball.style.background = pColor
          bojo.removeEventListener("mousemove", whileMove)
          bojo.removeEventListener("mouseup", afterUp)
          }) 
      })
      bojo.addEventListener("mouseup", function reload(){
        //sentQuestion();
        bojo.removeEventListener("mouseup",reload)
      })
  
    })
    //sentQuestion();
  }
  // // Wyłanie zapytania do serwera
  function sentQuestion() {
    ///Dziwny Blad
    loadPlayers()
    if (number_of_shooters_rev.current == 1) {
      //console.log('Wysyłanie wartości: ', bodyPart, technique, actionType, shooterPossition, gameMinute, firstShot);
      // Użyj backticksów zamiast zwykłych cudzysłowów
      fetch(`http://127.0.0.1:5000/get_model?shooter=${shooter}&goalkeeper=${goalkeeper}&defenders=${defenders}&strickers=${stricers}&bodyPart=${bodyPart}&technique=${technique}&actionType=${actionType}&shooterPossition=${shooterPossition}&gameMinute=${gameMinute}&shot_first_time=${firstShot}&shot_one_on_one=${oneOnOne}&shot_aerial_won=${afterAirDuel}&shot_open_goal=${openGoal}&shot_follows_dribble=${afterDribbling}&shot_redirect=${redirect}`).then(
        res => res.json()
      ).then(
        data => {
          setData(data);
          console.log(data);
          // Przenieś tę linię do środka bloku .then(), aby uniknąć błędów
          let eX = data.response;
          document.getElementById("ex").innerHTML = eX;
          //updateXGMeter(eX);
        }
      ).catch(error => {
        console.error('Błąd:', error);
      });
    } else {
      alert('Piłka nie jest obecnie na boisku.');
    } 
   
  }

  

  /* Funkcja dodająca listener do boiska*/
  function boiskoListener(ev){
    ev.preventDefault()
    var bojo = document.getElementById('footballField')
    var bnd = ev.target.getBoundingClientRect()

    // zapis od 0 - 1 w jakim procentowym miejscu boiska użytkownik kliknął
    var x = ((ev.clientX - bnd.left)/bojo.offsetWidth)*100;
    var y = ((ev.clientY - bnd.top)/bojo.offsetHeight)*100;

    var ball = document.createElement('div');
    ball.className = "football";
    ball.style.left = x + "%"
    ball.style.top = y + "%"
    
    //dodanie zawodnika do listy oraz punktu do mapy w zaleznosci od aktywnosci przycisku 1,2,3 lub 4
    
    if(active_bbt=="bbt1"){
      if(number_of_shooters_rev.current < 1 ){
        addPlayer(0,ball)
        bojo.appendChild(ball)
        ball.style.background = "#fc0303"
      }else{alert("mozesz dodac tylko jednego strzelca")}
    }else if(active_bbt == "bbt2"){
      if ( number_of_goalkeepers_rev.current < 1){
        addPlayer(1,ball)
        bojo.appendChild(ball)
        ball.style.background = "#03e7fc"   
      }else{alert("mozesz dodac tylko jednego bramkarza")}
    }else if(active_bbt == "bbt3"){
      if(number_of_defenders_rev.current < 10){
        addPlayer(2,ball);
        bojo.appendChild(ball)
        ball.style.background = "#0324fc"
    }else{alert("maksymalna liczba obroncow")}
   }else if(active_bbt == "bbt4"){
      if(number_of_strikers_rev.current < 10){
        addPlayer(3,ball);
        bojo.appendChild(ball)
        ball.style.background = "#fc6703"
      }else{alert("maksymalna liczba napastnikow")}
    }
}
// funkcja działą po utworzeniu komponentów, dodaje listenry do elementów

  useEffect(()=>{
      var footballField = document.getElementById('footballField');
      footballField.addEventListener('contextmenu', boiskoListener,false) 

      document.getElementById("bbt1").addEventListener("click",function(){
        changePlayer("bbt1")
      },false)
      document.getElementById("bbt2").addEventListener("click",function(){
        changePlayer("bbt2")
      },false)
      document.getElementById("bbt3").addEventListener("click",function(){
        changePlayer("bbt3")
      },false)
      document.getElementById("bbt4").addEventListener("click",function(){
        changePlayer("bbt4")
      },false)
    },[]);

  /*zwracany komponent zawierajacy boisko*/

  return (
<div>
<div className="container">
{/* Listy zwijane */}

<div className="top-bar" id = "top-bar">

<form className="dropdown" id = "bodyPartList">
            <select className="dropbtn"
            onChange={event => {setBodyPart(event.target.value);
                                //sentQuestion()
                              }}
            defaultValue={bodyPart}>
                    <option value = "Right Foot">Noga Prawa</option>
                    <option value = "Left Foot">Noga Lewa</option>
                    <option value = "Head">Głowa</option>
                    <option value = "Other" >Inna</option>
              </select>      
        </form>

        


        <form className="dropdown" id = "shootTypeList">
            <select className="dropbtn" onChange={event => {setTechnique(event.target.value);
                                                            //sentQuestion()
                                                            }}
            defaultValue = {technique}>
                    <option value="Normal"> Zwykły </option>
                    <option value = "Volley"> Wolej </option>
                    <option value = "Half Volley"> Półwolej </option>
                    <option value = "Lob"> Lob </option>     
                    <option value = "Diving Header"> Szczupak </option>
                    <option value = "Overhead Kick"> Kopnięcie z góry </option>
                    <option value = "Backheel"> Piętka </option>
                    </select>      
        </form>

     <form className="dropdown" id = "actionTypeList" onChange={event => {setActionType(event.target.value);
                                                                          //sentQuestion()
                                                                        }}
    defaultValue={technique}>
            <select className="dropbtn">
                    <option value = "Open Play"> Atak Pozycyjny </option>
                    <option value = "Free Kick"> Rzut Wolny </option>
                    <option value = "Penalty"> Rzut Karny </option>
                    <option value= "Corner"> Rzut Rozny </option>
            </select>      
        </form>
        <form className="dropdown" id = "possitionList"
        onChange={event => {setPossition(event.target.value);
                            //sentQuestion()
                          }}
        defaultValue={shooterPossition}>
            <select className="dropbtn">
            <option value="Right Center Forward">Prawy Środkowy Napastnik</option>
            <option value="Left Center Forward">Lewy Środkowy Napastnik</option>
            <option value="Secondary Striker">Drugi Napastnik</option>
            <option value="Center Forward">Środkowy Napastnik</option>
            <option value="Center Midfield">Środkowy Pomocnik</option>
            <option value="Left Center Midfield">Lewy Środkowy Pomocnik</option>
            <option value="Right Center Midfield">Prawy Środkowy Pomocnik</option>
            <option value="Left Midfield">Lewy Pomocnik</option>
            <option value="Center Attacking Midfield">Środkowy Ofensywny Pomocnik</option>
            <option value="Left Defensive Midfield">Lewy Defensywny Pomocnik</option>
            <option value="Left Attacking Midfield">Lewy Ofensywny Pomocnik</option>
            <option value="Right Attacking Midfield">Prawy Ofensywny Pomocnik</option>
            <option value="Right Defensive Midfield">Prawy Defensywny Pomocnik</option>
            <option value="Center Defensive Midfield">Środkowy Defensywny Pomocnik</option>
            <option value="Right Midfield">Prawy Pomocnik</option>
            <option value="Left Wing Back">Lewy Pomocnik Skrzydłowy</option>
            <option value="Left Wing">Lewe Skrzydło</option>
            <option value="Right Wing">Prawe Skrzydło</option>
            <option value="Right Wing Back">Prawy Pomocnik Skrzydłowy</option>
            <option value="Center Back">Środkowy Obrońca</option>
            <option value="Left Center Back">Lewy Środkowy Obrońca</option>
            <option value="Right Center Back">Prawy Środkowy Obrońca</option>
            <option value="Right Back">Prawy Obrońca</option>
            <option value="Left Back">Lewy Obrońca</option>
            <option value="Goalkeeper">Bramkarz</option>
                    </select>      
        </form>

        <div className="cho-minute">
                <label htmlFor="gameMinute" className="label-large"></label>
                <input 
                    type="number" 
                    id="gameMinute" 
                    name="gameMinute" 
                    value={gameMinute} 
                    onChange={handleMinuteChange} 
                    min="0" 
                    placeholder="Wpisz minutę"
                />
            </div>
          </div>
      
      <div className="main-content" id = "field">   
          <div className="player-list" id = "list" >
            
          </div>
          <div className="field" id="footballField">   
              <div className="field-pic"></div>
              
        </div>
<div className="rightMenu">
    <div className="xg-meter">
      <b id="ex" className="xg-value">0</b>
    </div>

 <div className="additional-parameters">
  <h3>Parametry strzału</h3>
  <label>
    <input type="checkbox" id="firstShot" onChange={handleFirstShotChange} />
    Pierwszy w meczu
  </label>
  <label>
    <input type="checkbox" id="oneOnOne" onChange={handleOneOnOneChange} />
    Akcja sam na sam
  </label>
  <label>
    <input type="checkbox" id="afterAirDuel" onChange={handleAfterAirDuelChange} />
    Po pojedynku powietrznym
  </label>
  <label>
    <input type="checkbox" id="openGoal" onChange={handleOpenGoalChange} />
    Na pustą bramkę
  </label>
  <label>
    <input type="checkbox" id="afterDribbling" onChange={handleAfterDribblingChange} />
    Poprzedzony dryblingiem
  </label>
  <label>
    <input type="checkbox" id="redirect" onChange={handleRedirectChange} />
    Rykoszet
  </label>
</div>
</div>
</div>
<div className="bottom-bar">

      <button className="cho-shooter" id = "bbt1">Strzelec</button>
      <button className="cho-goalkeeper" id = "bbt2">Bramkarz</button>
      <button className="cho-defence" id = "bbt3">Broniący</button>
      <button className="cho-atack"  id = "bbt4">Atakujący</button>
      <button className="reset-button" onClick={resetField}>Reset</button>  
      <button className="info-button" onClick={sentQuestion}>xG</button>
     </div>
          </div>   
   </div>
  );
};

export default Hero;
