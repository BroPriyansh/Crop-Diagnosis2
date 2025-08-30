import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword } from "firebase/auth";
const auth = getAuth();

const userCred = await createUserWithEmailAndPassword(auth, email, password);
const token = await userCred.user.getIdToken();
await fetch("http://127.0.0.1:8004/sign-up", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${token}`
  },
  body: JSON.stringify({
    uid: userCred.user.uid,
    email: userCred.user.email,
    display_name: "John Doe"
  })
});



