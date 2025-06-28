import React from 'react';
// import './Sidebar.css'; // Artık global index.css ile stiller geliyor

const Sidebar = () => (
  <aside className="sidebar">
    <div className="logo">Dursun</div>
    <nav>
      <button className="active" style={{background:'none',border:'none',color:'inherit',textAlign:'left',padding:'12px 24px',font:'inherit',cursor:'pointer'}}>Dashboard</button>
      <button style={{background:'none',border:'none',color:'inherit',textAlign:'left',padding:'12px 24px',font:'inherit',cursor:'pointer'}}>Video</button>
      <button style={{background:'none',border:'none',color:'inherit',textAlign:'left',padding:'12px 24px',font:'inherit',cursor:'pointer'}}>Durum</button>
      <button style={{background:'none',border:'none',color:'inherit',textAlign:'left',padding:'12px 24px',font:'inherit',cursor:'pointer'}}>Kontroller</button>
      <button style={{background:'none',border:'none',color:'inherit',textAlign:'left',padding:'12px 24px',font:'inherit',cursor:'pointer'}}>Loglar</button>
      <button style={{background:'none',border:'none',color:'inherit',textAlign:'left',padding:'12px 24px',font:'inherit',cursor:'pointer'}}>Ayarlar</button>
    </nav>
  </aside>
);

export default Sidebar;
