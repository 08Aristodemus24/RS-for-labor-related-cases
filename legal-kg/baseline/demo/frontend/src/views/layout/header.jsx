import React, {useState} from "react";
import SideMenu from './sideMenu';
import './css/header.css'
import ReactCSSTransitionGroup from 'react-addons-css-transition-group'
import {useGlobalState} from "../../state";
import RequestManager from "../../utils/requestManager";
import Settings from "../../components/Settings";

function Header() {
    const [availableItems] = useGlobalState('availableItems');
    const [subgraph, setSubgraph] = useGlobalState('subgraph');
    const [selectedNodes, setSelectedNodes] = useGlobalState('selectedNodes');
    const [data, setData] = useGlobalState('data');
    const [options, setOptions] = useGlobalState('options');
    const [sideMenuOpen, setSideMenuOpen] = useState(true);
    const [optionsOpen, setOptionsOpen] = useState(false);

    const changeSubgraph = (event) => {
        const subgraphID = parseInt(event.target.value);
        setSelectedNodes([]);
        setSubgraph(subgraphID);

        RequestManager.get(`/subgraphs/${subgraphID}`)
            .then((res) => RequestManager.validateResponse(res))
            .then((data) => {
                setData({
                    links: data.graph.links,
                    nodes: data.graph.nodes,
                    duplicateMapping: data.duplicate_mapping,
                    mapping: data.mapping,
                    groupedPersonAttributes: data.grouped_person_attributes
                });
            })
            .catch((err) => {
                console.error(err);
            })
    };

    const changeSideMenuWidth = () => {
        let width = options.sideMenuWidth;
        let opacity = options.sideMenuOpacity;

        if (width === 80) {
            width = 50;
            opacity = 0.75;
        } else {
            width += 15;
            opacity += 0.05;
        }

        const newOptions = {
            ...options,
            sideMenuWidth: width,
            sideMenuOpacity: opacity
        };

        setOptions(newOptions);
    };

    return (
        <div>
            <ReactCSSTransitionGroup transitionName="slider" transitionEnterTimeout={300}
                                     transitionLeaveTimeout={300}>
                {
                    sideMenuOpen ? <SideMenu/> : undefined
                }
            </ReactCSSTransitionGroup>
            <ReactCSSTransitionGroup transitionName="modal" transitionEnterTimeout={300}
                                     transitionLeaveTimeout={300}>
                {
                    optionsOpen ? <Settings close={() => setOptionsOpen(!optionsOpen)}/> : undefined
                }
            </ReactCSSTransitionGroup>
            <div className="navBarStyle">
                <span className="headerWatsonTextStyle"><strong id="title">IBM </strong>Node Similarity Detection - Demo</span>

                <div className="rightNavBarStyle">
                    <a onClick={() => setSideMenuOpen(!sideMenuOpen)}><img className="menuIconStyle" alt="Menu"
                                                                           src={require('./assets/menu.png')}/></a>
                </div>
                {
                    options.sideMenuResizeable ?
                        <div className="rightNavBarStyle">
                            <a onClick={changeSideMenuWidth}><img className="menuIconStyle" alt="Expand"
                                                                  src={require('./assets/expand.png')}/></a>
                        </div> : undefined
                }
                <div className="rightNavBarStyle">
                    <a onClick={() => setOptionsOpen(!optionsOpen)}><img className="menuIconStyle" alt="Login"
                                                                         src={require('./assets/settings.png')}/></a>
                </div>
                {
                    options.showDebuggingMode ?
                        <div className="dropdownWrapperStyle">
                            <select value={subgraph} onChange={changeSubgraph} className="select-css">
                                {
                                    availableItems.subgraphs.map((e, i) => <option key={i}
                                                                                   value={e}>Subgraph {e}</option>)
                                }
                            </select>
                        </div> : undefined
                }
            </div>
        </div>
    );
}

export default Header;
