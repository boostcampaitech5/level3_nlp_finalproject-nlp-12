import axios from 'axios';
import React, { useEffect, useState } from 'react';
import packageJson from '../package.json'

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSpinner, faArrowTurnDown, faEraser } from '@fortawesome/free-solid-svg-icons'

import './App.css';
import Message from './message';

axios.defaults.baseURL = packageJson.proxy;
axios.defaults.withCredentials = true;

function App() {

  const [msg_text, setMsgText] = useState('')     // input박스에 입력된 유저 메시지
  const [user_id , setUserID ] = useState('')     // cookie에 저장된 유저 해시(최초 접속시 비어있음)
  const [msg_list, setMsgList] = useState(null);  // 전체 메시지 히스토리
  const [showing , setShowing] = useState(false); // 페이지 로딩중 여부

  const getMsg = async () => {
    await axios.get(
      '/'
      ).then(response =>{
        setMsgList(response.data.msg_list);
        setUserID(response.data.user_id);
        setShowing(true);
    })
    .catch(error => {
      console.error('error', error);
      setMsgList('error');
    });
  }

  const postMsg = async () => {
    if (msg_text) {
      setMsgList(prev => [...prev, {msg_text: msg_text, bot: false}]);
      var current_msg_text = msg_text;
      setMsgText('')
      await axios.post(
            '/',
            {
                user_id: user_id,
                msg_text: current_msg_text,
                bot: false
            },
            {
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json"
                },
            }
        ).then((response) => {
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
    axios.delete(
      '/'
    ).then((response) =>{
      console.log(response)
      setMsgText('');
      getMsg();
    }
    ).catch((error) => {
      console.log(error)
    });
  };

  useEffect(() => {
    getMsg()
  }, []);

  if (showing){
    return (
      <div className="App" key='App'>
        <header className="App-header">
          <div>
            {msg_list.map(msg =>(
              <Message msg={msg}/>
          ))}</div>
          <div className="msg_input" key='msg_input'>
                <input
                    type='text'
                    required={true}
                    placeholder='메시지를 입력해보세요.'
                    autoFocus={true}
                    onChange={onInputValChange}
                    onKeyUp={onInputKey}
                    value={msg_text}
                    style={{border: 'none', outline: 'none', width: '60vw', background: 'none'}}
                />
                <button
                    onClick={postMsg}
                    style={{border: 'none', outline: 'none', background: 'none'}}>
                    <FontAwesomeIcon icon={faArrowTurnDown} rotation={90} style={{color: "#000000",}} />
                </button>
            </div>
                {/* TODO: add callback func */}
                {/* TODO: add hover animation */}
                <FontAwesomeIcon className='Eraser' icon={faEraser} style={{color: "#000000", padding: '50px'}} onClick={deleteMsg} />
        파이렝...
        </header>
      </div>
    );
  } else{
    return(
      <div className="App" key='App'>
        <header className="App-header">
          <FontAwesomeIcon icon={faSpinner} spinPulse size='3x' style={{color: "#dedede",}} />
        </header>
      </div>
    );
  };
};

export default App;