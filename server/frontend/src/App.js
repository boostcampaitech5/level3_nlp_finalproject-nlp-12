import axios from 'axios';
import React, { useEffect, useState, useRef } from 'react';
import packageJson from '../package.json';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowTurnDown, faEraser, faEllipsis } from '@fortawesome/free-solid-svg-icons';

import './App.css';
import Message from './message';

axios.defaults.baseURL = packageJson.proxy;
axios.defaults.withCredentials = true;

function App() {

  const [msg_text,    setMsgText   ] = useState('');     // input박스에 입력된 유저 메시지
  const [user_id ,    setUserID    ] = useState('');     // cookie에 저장된 유저 해시(최초 접속시 비어있음)
  const [msg_list,    setMsgList   ] = useState(null);   // 전체 메시지 히스토리
  const [showing ,    setShowing   ] = useState(false);  // 페이지 로딩중 여부
  const [generating,  setGenerating] = useState(false);  // 메시지 생성중 스피너

  const focusRef = useRef(null);

  const getMsg = async () => {
    await axios.get(
      '/'
    ).then((response) =>{
        console.log(response);
        setMsgList(response.data.msg_list);
        setUserID(response.data.user_id);
        setShowing(true);
    }).catch((error) => {
      console.error('error', error);
      setMsgList('error');
    });
  }

  const postMsg = async () => {
    if (msg_text) {
      setMsgList(prev => [...prev, {msg_text: msg_text, bot: false}]);
      var current_msg_text = msg_text;
      setMsgText('');
      setGenerating(true);
      await axios.post(
        '/', // path
        { // data
            user_id: user_id,
            msg_text: current_msg_text,
            bot: false
        },
        { // config
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json"
            },
        }
      ).then((response) => {
        setGenerating(false);
        setMsgList(prev => [...prev, response.data]);
      }).catch((error) => {
        console.log(error);
      });
    }
  };

  const onInputValChange = (e) => {
    setMsgText(e.target.value);
  };

  const onInputKey = async (e) => {
    if (e.code==='Enter' && msg_text !== ''){
      await postMsg();
    }
  }

  const deleteMsg = () => { // reset버튼을 위한 callback 함수 프로토타입
    setShowing(false);
    axios.delete(
      '/'
    ).then((response) =>{
      console.log(response);
      setMsgText('');
      getMsg();
    }
    ).catch((error) => {
      console.log(error);
    });
  };

  useEffect(() => {
    getMsg();
  }, []);

  useEffect(() => {
    focusRef.current?.scrollIntoView({ behavior: 'smooth' , block: 'center' });
  }, [msg_list, showing, generating]);

  if (showing){
    return (
      <div className="App" key='App'>
        <header className="header">
          <div style={{width:'60vw', padding:'10px', alignItems:'center'}}>
            <img src='./mary.svg' style={{width:'5vw', padding:'10px'}} />
          </div>
        </header>
        <div className="App-header" style={{minHeight:'200vh'}}>
          <div style={{width:'60vw'}}>
            {msg_list.map(msg =>(
              <Message msg={msg}/>
            ))}
            { generating === true &&
              <div align="left" style={{width: '60vw', padding: '20px'}}>
                <FontAwesomeIcon className='generation' icon={faEllipsis} fade size='xs' style={{color: "#000000"}} />
              </div>
            }
          <div ref={focusRef}></div>
          <div className="msg_input" key='msg_input' style={{width: '60vw', padding: '10px', alignContent:'center'}}>
                <textarea
                    required={true}
                    placeholder='마리에게 말을 걸어 보세요(최대 200자).'
                    autoFocus={true}
                    onChange={onInputValChange}
                    onKeyUp={onInputKey}
                    maxLength={200}
                    rows={5}
                    value={msg_text}
                    style={{border: 'none', outline: 'none', width: '59vw', marginTop:'5vh', background: 'none', padding: '0px'}}
                />
                <button
                    onClick={postMsg}
                    style={{width: '1vw', padding: '0px', border: 'none', outline: 'none', background: 'none'}}>
                    <FontAwesomeIcon icon={faArrowTurnDown} rotation={90} style={{color: "#000000"}} />
                </button>
            </div>
          </div>
            <div className='bottom'>
              <div className='tooltip' style={{padding: '20px'}}>
                  {/* TODO: add hover animation */}
                  <FontAwesomeIcon className='Eraser'
                                  icon={faEraser}
                                  style={{color: "#000000"}}
                                  onClick={deleteMsg} /><br/>
                  <span className='tooltiptext'>대화 지우기</span>           
              </div>
            </div>
            {/* <div className='footer'>
              MyMary by Smoked_Salmons<br/>
              boostcamp AI Tech 5th
            </div> */}
        </div>
      </div>
    );
  } else{
    return(
      <div className="App" key='App'>
        <div className="App-header" style={{height:'100vh'}}>
          {/* <FontAwesomeIcon ref={focusRef} icon={faSpinner} spinPulse size='3x' style={{color: "#dedede"}} /> */}
          <img ref={focusRef} src='mary.svg' className='op2' width='200vw'/>
        </div>
      </div>
    );
  };
};

export default App;