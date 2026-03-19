package com.swimming;

@service 
public class SessionService {

    this.SessionRepository = SessionRepository;

    public List<Sessions> getAllSessions ()
    {
        return SessionRepository.findall();
    }

    public Session getSessionbyID(Long id)
    {
        if(session ==id)
        return Session.Repository.findById(id)
    }

    public Session updateSession (Long id,Session session_details)
    {
        

        
    }





    public void DeleteSession(Long id)
    {
        SessionRepository.deleteById(id);
    }
}
