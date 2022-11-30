import React from 'react';
import ReactDOM from 'react-dom';
import './index.scss';
import * as serviceWorker from './serviceWorker';

import {BrowserRouter, Redirect} from "react-router-dom";
import {Route, Switch} from "react-router";

import Demo from "./views/demo";
import Header from "./views/layout/header";
import Body from "./views/layout/body";

import {GlobalStateProvider} from './state';

const AppRoute = ({component: Component, defaultLayout, ...rest}) => {
    return (
        <Route {...rest}
               render={
                   (props) => {
                       if (defaultLayout) {
                           return (
                               <GlobalStateProvider>
                                   <Header/>
                                   <Body>
                                       <Component {...props} />
                                   </Body>
                               </GlobalStateProvider>
                           );
                       }

                       return <Component {...props} />;
                   }
               }
        />
    );
};

ReactDOM.render((
    <BrowserRouter>
        <Switch>
            <AppRoute exact path='/' defaultLayout={true} component={Demo}/>
            <Route render={props => (<Redirect to={{pathname: '/', state: {from: props.location}}}/>)}/>
        </Switch>
    </BrowserRouter>
), document.getElementById('root'));

serviceWorker.unregister();
