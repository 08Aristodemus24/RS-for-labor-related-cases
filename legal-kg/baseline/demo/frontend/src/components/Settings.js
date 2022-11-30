import * as React from "react";
import "./scss/Settings.scss";
import {useGlobalState} from "../state";

function Settings(props) {
    const [options, setOptions] = useGlobalState('options');

    const handleChange = (event, checkbox, number, object, secondOrderObject) => {
        let isUndefined = false;
        if (event.target.value === "") {
            isUndefined = true;
        }

        if (secondOrderObject) {
            return setOptions({
                ...options,
                [object]: {
                    ...options[object],
                    [secondOrderObject]: {
                        ...options[object][secondOrderObject],
                        [event.target.name]: checkbox ? event.target.checked : number ? parseInt(event.target.value) : isUndefined ? undefined : event.target.value
                    }
                }
            });
        }

        if (object) {
            return setOptions({
                ...options,
                [object]: {
                    ...options[object],
                    [event.target.name]: checkbox ? event.target.checked : number ? parseInt(event.target.value) : isUndefined ? undefined : event.target.value
                }
            });
        }

        setOptions({
            ...options,
            [event.target.name]: checkbox ? event.target.checked : number ? parseInt(event.target.value) : isUndefined ? undefined : event.target.value
        });
    };

    const close = () => {
        localStorage.setItem('options', JSON.stringify(options));
        props.close();
    };

    return (
        <div className="dataModalContainer">
            <div className="dataModalContentContainer">
                <div className="headerContainer">
                    <div className="titleContainer">
                        <h2 className="title">Options</h2>
                    </div>
                </div>
                <div className="dataModalContent">
                    <div>
                        <p>General</p>
                        <hr/>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showDebuggingMode"
                                       defaultChecked={options.showDebuggingMode}
                                       onChange={(e) => handleChange(e, true, false)}/>
                                <span className="slider round"/>
                            </label>
                            <span>Advanced Mode (for debugging)</span>
                        </div>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="sideMenuResizeable"
                                       defaultChecked={options.sideMenuResizeable}
                                       onChange={(e) => handleChange(e, true, false)}/>/>
                                <span className="slider round"/>
                            </label>
                            <span>Sidemenu resizeable</span>
                        </div>
                        <br/>
                        <p>Graph Visualization</p>
                        <hr/>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="active"
                                       defaultChecked={options.duplicateNodeHighlighting.active}
                                       onChange={(e) => handleChange(e, true, false, "duplicateNodeHighlighting")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Highlight duplicate nodes (ground truth)</span>
                        </div>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="active"
                                       defaultChecked={options.duplicateLinkHighlighting.active}
                                       onChange={(e) => handleChange(e, true, false, "duplicateLinkHighlighting")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Highlight links between duplicate nodes</span>
                        </div>
                        {
                            options.duplicateLinkHighlighting.active ?
                                <div>
                                    <div className="switchContainer indented">
                                        <input className="inputValue" name="color"
                                               defaultValue={options.duplicateLinkHighlighting.color}
                                               onChange={(e) => handleChange(e, false, false, "duplicateLinkHighlighting")}
                                               type="text"/>
                                        <span>Highlight Color</span>
                                    </div>
                                    <div className="switchContainer indented">
                                        <input className="inputValue" name="connected"
                                               defaultValue={options.duplicateLinkHighlighting.strokeWidth.connected}
                                               onChange={(e) => handleChange(e, false, true, "duplicateLinkHighlighting",
                                                   "strokeWidth")} type="number" min={0}/>
                                        <span>Stroke Width - Already connected components</span>
                                    </div>
                                    <div className="switchContainer indented">
                                        <input className="inputValue" name="notConnected"
                                               defaultValue={options.duplicateLinkHighlighting.strokeWidth.notConnected}
                                               onChange={(e) => handleChange(e, false, true, "duplicateLinkHighlighting",
                                                   "strokeWidth")} type="number" min={0}/>
                                        <span>Stroke Width - Not connected components</span>
                                    </div>
                                </div> : undefined
                        }
                        <div className="switchContainer">
                            <input className="inputValue" name="subgraph" min={0} max={1000}
                                   defaultValue={options.default.subgraph}
                                   onChange={(e) => handleChange(e, false, true, "default")} type="number"/>
                            <span>Default Subgraph ID</span>
                        </div>
                        <div className="switchContainer">
                            <input className="inputValue" name="featureSet"
                                   defaultValue={options.default.featureSet}
                                   onChange={(e) => handleChange(e, false, false, "default")} type="text"/>
                            <span>Default Feature Set Name</span>
                        </div>
                        <br/>
                        <p>Attribute Comparison</p>
                        <hr/>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showOnlyAvailableAttributes"
                                       defaultChecked={options.attributeComparison.showOnlyAvailableAttributes}
                                       onChange={(e) => handleChange(e, true, false, "attributeComparison")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Show only available attributes</span>
                        </div>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showSubject"
                                       defaultChecked={options.attributeComparison.showSubject}
                                       onChange={(e) => handleChange(e, true, false, "attributeComparison")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Include subject attribute</span>
                        </div>
                        <div className="switchContainer">
                            <label className="switch">
                                <input type="checkbox" name="showMatching"
                                       defaultChecked={options.attributeComparison.showMatching}
                                       onChange={(e) => handleChange(e, true, false, "attributeComparison")}/>
                                <span className="slider round"/>
                            </label>
                            <span>Highlight string matching between attributes</span>
                        </div>
                        {
                            options.attributeComparison.showMatching ?
                                <div>
                                    <div className="switchContainer indented">
                                        <input className="inputValue" name="color"
                                               defaultValue={options.attributeComparison.matchingStyle.color}
                                               onChange={(e) => handleChange(e, false, false, "attributeComparison", "matchingStyle")}
                                               type="text"/>
                                        <span>Highlight Color</span>
                                    </div>
                                </div> : undefined
                        }
                    </div>
                </div>
                <div className="buttonContainer">
                    <button onClick={close} className="button">
                        Save and close
                    </button>
                </div>
            </div>
        </div>
    );
}


export default Settings;