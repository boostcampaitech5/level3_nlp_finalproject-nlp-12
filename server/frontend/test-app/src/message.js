import React from "react";
import randomString from './randomString';

function Message({msg}) {
    console.log(msg)
    var component_id = randomString(8)
    if (msg.bot) {
        return (
            <div align="left" key={component_id} style={{width: '60vw', padding: '20px', fontWeight:800}}>
                {msg.msg_text}
            </div>
        );
    } else{
        return (
            <div align="right" key={component_id} style={{width: '60vw', padding: '20px', color:'gray', fontSize:'0.8em'}}>
                {msg.msg_text}
            </div>
        );
    };
};

export default React.memo(Message);